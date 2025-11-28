# vllm-ascend CI

## overview

本目录包含了vllm-ascend项目的所有自动化流水线源代码，主要服务于本项目的pr集成验证（lint、 ut、 e2e）、镜像/二进制制品构建/分发、定时触发任务（功能、性能、精度），社区基础功能集成（format_pr_body/labeler/auto bot..）

## 命名标准

按照以下几个标准来命名：

```shell
<triggering_type>_<test_type>_<func_type>.yaml
```

1. 这个workflow是如何被触发的：包括pr触发、定时任务触发、workflow_call、label触发、tag触发



## 目录结构概览


```shell
.github
├── Dockerfile.buildwheel
├── Dockerfile.nightly.a2
├── Dockerfile.nightly.a3
├── PULL_REQUEST_TEMPLATE.md
├── actionlint.yaml
├── dependabot.yml
├── format_pr_body.sh
├── labeler.yml
└── workflows
    ├── README.md
    ├── _e2e_nightly_multi_node.yaml
    ├── _e2e_nightly_single_node.yaml
    ├── _e2e_nightly_single_node_models.yaml
    ├── _e2e_test.yaml
    ├── _nightly_image_build.yaml
    ├── format_pr_body.yaml
    ├── image_310p_openeuler.yml
    ├── image_310p_ubuntu.yml
    ├── image_a3_openeuler.yml
    ├── image_a3_ubuntu.yml
    ├── image_openeuler.yml
    ├── image_ubuntu.yml
    ├── label_merge_conflict.yml
    ├── labeler.yml
    ├── matchers
    │   ├── actionlint.json
    │   ├── mypy.json
    │   └── ruff.json
    ├── nightly_benchmarks.yaml
    ├── pre-commit.yml
    ├── release_code.yml
    ├── release_whl.yml
    ├── reminder_comment.yml
    ├── vllm_ascend_doctest.yaml
    ├── vllm_ascend_test_310p.yaml
    ├── vllm_ascend_test_full_vllm_main.yaml
    ├── vllm_ascend_test_nightly_a2.yaml
    ├── vllm_ascend_test_nightly_a3.yaml
    ├── vllm_ascend_test_pr_full.yaml
    ├── vllm_ascend_test_pr_light.yaml
    └── vllm_ascend_test_report.yaml
```


referring to [github action](https://docs.github.com/en/actions/get-started/quickstart)

