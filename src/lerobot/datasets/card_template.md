---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
# prettier-ignore
{{card_data}}
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

{% if repo_id is defined and repo_id %}
<a class="flex" href="https://huggingface.co/spaces/lerobot/visualize_dataset?path={{ repo_id }}">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl.svg"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl-dark.svg"/>
</a>
{% endif %}

## Dataset Description

{{ dataset_description | default("", true) }}

- **Homepage:** {{ url | default("[More Information Needed]", true)}}
- **Paper:** {{ paper | default("[More Information Needed]", true)}}
- **License:** {{ license | default("[More Information Needed]", true)}}

## Dataset Structure

{{ dataset_structure | default("[More Information Needed]", true)}}

## Citation

**BibTeX:**

```bibtex
@misc{vertax2026lerobotxense,
    author = {vertax42 and Xense Robotics Team},
    title = {LeRobot-Xense: LeRobot with Xense Tactile Robotics Support},
    howpublished = {\url{https://github.com/Vertax42/lerobot-xense}},
    year = {2026}
}{% if citation_bibtex is defined and citation_bibtex %}

{{ citation_bibtex }}{% endif %}
```
