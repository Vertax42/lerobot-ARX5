from itertools import count
from time import sleep
from threading import Event
import atexit
import argparse
from urllib.parse import urljoin


from abc import ABC, abstractmethod
import hashlib
import base64
import requests
from urllib.parse import urljoin
from http import HTTPStatus


class FrankaClient(ABC):
    """
    Low-level client for Franka robot web interface authentication and control.
    
    This abstract base class handles login, logout, control token management,
    and shutdown operations via the robot's HTTP/HTTPS API. Used internally
    by FrankaLockUnlock for brake control and FCI activation.
    
    Attributes:
        _hostname (str): Robot hostname with protocol (http:// or https://)
        _username (str): Admin username
        _password (str): Admin password (SHA256 encoded)
        _logged_in (bool): Current login state
        _token (str): Active control token
        _token_id (int): Active control token ID
        
    Note:
        This is an abstract class. Use FrankaLockUnlock for actual robot control.
        
    Caveats:
        - Only one user can have control token at a time
        - SSL verification disabled by default (self-signed certs)
        - Must login before requesting control token
        - Token expires if not used within timeout period
    """
    def __init__(self, hostname: str, username: str, password: str, protocol: str = 'http'):
        """
        Initialize Franka client with credentials.
        
        Args:
            hostname (str): Robot IP or hostname (e.g., "172.16.0.2")
            username (str): Admin username (typically "admin")
            password (str): Admin password
            protocol (str): "http" or "https" (default: "http")
            
        Note:
            Password is automatically SHA256 encoded using Franka's format.
        """
        requests.packages.urllib3.disable_warnings()
        self._session = requests.Session()
        self._session.verify = False
        self._hostname = f'{protocol}://{hostname}'
        self._username = username
        self._password = password
        self._logged_in = False
        self._token = None
        self._token_id = None

    @staticmethod
    def _encode_password(username, password):
        bs = ','.join([str(b) for b in hashlib.sha256((f'{password}#{username}@franka').encode('utf-8')).digest()])
        return base64.encodebytes(bs.encode('utf-8')).decode('utf-8')

    def _login(self):
        print("Logging in...")
        if self._logged_in:
            print("Already logged in.")
            return
        login = self._session.post(urljoin(self._hostname, '/admin/api/login'), \
                                           json={'login': self._username, \
                                                 'password': self._encode_password(self._username, self._password)})
        assert login.status_code == HTTPStatus.OK, "Error logging in."
        self._session.cookies.set('authorization', login.text)
        self._logged_in = True
        print("Successfully logged in.")

    def _logout(self):
        print("Logging out...")
        assert self._logged_in
        logout = self._session.post(urljoin(self._hostname, '/admin/api/logout'))
        assert logout.status_code == HTTPStatus.OK, "Error logging out"
        self._session.cookies.clear()
        self._logged_in = False
        print("Successfully logged out.")

    def _shutdown(self):
        print("Shutting down...")
        assert self._is_active_token(), "Cannot shutdown without an active control token."
        try:
            self._session.post(urljoin(self._hostname, '/admin/api/shutdown'), json={'token': self._token})
        except requests.exceptions.RequestException as _:
            # Sometimes, the server can shut down before sending a complete response, possibly raising an exception.
            # Anyways, the server has still received the request, thus the robot shutdown procedure will start.
            # So, we can ignore the cases when these exceptions are raised.
            pass
        finally:
            print("The robot is shutting down. Please wait for the yellow lights to turn off, then switch the control box off.")

    def _get_active_token_id(self):
        token_query = self._session.get(urljoin(self._hostname, '/admin/api/control-token'))
        assert token_query.status_code == HTTPStatus.OK, "Error getting control token status."
        json = token_query.json()
        return None if json['activeToken'] is None else json['activeToken']['id']

    def _is_active_token(self):
        active_token_id = self._get_active_token_id()
        return active_token_id is None or active_token_id == self._token_id

    def _request_token(self, physically=False):
        print("Requesting a control token...")
        if self._token is not None:
            assert self._token_id is not None
            print("Already having a control token.")
            return
        token_request = self._session.post(urljoin(self._hostname, f'/admin/api/control-token/request{"?force" if physically else ""}'), \
                                           json={'requestedBy': self._username})
        assert token_request.status_code == HTTPStatus.OK, "Error requesting control token."
        json = token_request.json()
        self._token = json['token']
        self._token_id = json['id']
        print(f'Received control token is {self._token} with id {self._token_id}.')

    def _release_token(self):
        print("Releasing control token...")
        token_delete = self._session.delete(urljoin(self._hostname, '/admin/api/control-token'), \
                                                    json={'token': self._token})
        assert token_delete.status_code == 200, "Error releasing control token."
        self._token = None
        self._token_id = None
        print("Successfully released control token.")

    @abstractmethod
    def run(self) -> None:
        pass


class FrankaLockUnlock(FrankaClient):
    """
    High-level client for Franka robot brake control and FCI activation.
    
    This class provides methods to lock/unlock robot brakes, activate FCI
    (Franka Control Interface), and manage control tokens. Essential for
    preparing the robot before using RobotInterface.
    
    Typical workflow:
    1. Create FrankaLockUnlock instance
    2. Call run(unlock=True, fci=True, persistent=True)
    3. Use RobotInterface for control
    4. Cleanup automatically handles relock and logout
    
    Examples:
        Unlock and activate FCI:
        >>> client = FrankaLockUnlock("172.16.0.2", "admin", "admin")
        >>> client.run(unlock=True, fci=True, persistent=True)
        
        Lock robot:
        >>> client.run(unlock=False)
        
        Request physical access (requires button press):
        >>> client.run(unlock=True, request=True, wait=True)
        
    Caveats:
        - Must be called before first use of RobotInterface
        - Use persistent=True to keep token for multiple scripts
        - relock=True automatically locks brakes on exit
        - Physical access requires pressing button on robot
    """
    def __init__(self, hostname: str, username: str, password: str, protocol: str = 'https', relock: bool = False):
        """
        Initialize lock/unlock client.
        
        Args:
            hostname (str): Robot IP address (e.g., "172.16.0.2")
            username (str): Admin username (default: "admin")
            password (str): Admin password
            protocol (str): "http" or "https" (default: "https")
            relock (bool): Automatically lock brakes on exit (default: False)
            
        Note:
            - Cleanup handler registered automatically via atexit
            - relock=True is useful for safety but may be unwanted in scripts
        """
        super().__init__(hostname, username, password, protocol=protocol)
        self._relock = relock
        atexit.register(self._cleanup)

    def _cleanup(self):
        print("Cleaning up...")
        if self._relock:
            self.run(unlock=False)
        if self._token is not None or self._token_id is not None:
            self._release_token()
        if self._logged_in:
            self._logout()
        print("Successfully cleaned up.")

    def _activate_fci(self):
        print("Activating FCI...")
        fci_request = self._session.post(urljoin(self._hostname, f'/admin/api/control-token/fci'), \
                                         json={'token': self._token})
        assert fci_request.status_code == 200, "Error activating FCI."
        print("Successfully activated FCI.")

    def _home_gripper(self):
        print("Homing the gripper...")
        action = self._session.post(urljoin(self._hostname, f'/desk/api/gripper/homing'), \
                                    headers={'X-Control-Token': self._token})
        assert action.status_code == 200, "Error homing gripper."
        print(f'Successfully homed the gripper.')

    def _lock_unlock(self, unlock: bool, force: bool = False):
        print(f'{"Unlocking" if unlock else "Locking"} the robot...')
        action = self._session.post(urljoin(self._hostname, f'/desk/api/robot/{"open" if unlock else "close"}-brakes'), \
                                    files={'force': force},
                                    headers={'X-Control-Token': self._token})
        assert action.status_code == 200, "Error requesting brake open/close action."
        print(f'Successfully {"unlocked" if unlock else "locked"} the robot.')

    def run(self, unlock: bool = False, force: bool = False, wait: bool = False, request: bool = False, persistent: bool = False, fci: bool = False, home: bool = False) -> None:
        assert not request or wait, "Requesting control without waiting for obtaining control is not supported."
        assert not fci or unlock, "Activating FCI without unlocking is not possible."
        assert not fci or persistent, "Activating FCI without persistence is not possible."
        assert not home or unlock, "Homing the gripper without unlocking is not possible."
        self._login()
        try:
            assert self._token is not None or self._get_active_token_id() is None or wait, "Error requesting control, the robot is currently in use."
            while True:
                self._request_token(physically=request)
                try:
                    # Consider the timeout of 20 s for requesting physical access to the robot
                    for _ in range(20) if request else count():
                        if (not wait and not request) or self._is_active_token():
                            print('Successfully acquired control over the robot.')
                            self._lock_unlock(unlock=unlock)
                            if home:
                                self._home_gripper()
                            if fci:
                                self._activate_fci()
                            return
                        if request:
                            print('Please press the button with the (blue) circle on the robot to confirm physical access.')
                        elif wait:
                            print('Please confirm the request message in the web interface on the logged in user.')
                        sleep(1)
                    # In case physical access was not confirmed, try again
                    self._release_token()
                finally:
                    if not persistent:
                        self._release_token()
        finally:
            if not persistent:
                self._logout()

