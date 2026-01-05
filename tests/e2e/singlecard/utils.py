from dataclasses import dataclass, field
from typing import Optional

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal


@dataclass(frozen=True)
class LLMTestCase:
    model: str
    prompts: list[str]
    golden_answers: list[str]
    quantization: Optional[str] = None
    sampling_params: SamplingParams = field(
        default_factory=lambda: SamplingParams(
            max_tokens=32,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            n=1,
        ))


def gen_and_valid(runner_kwargs: dict, prompts: list[str],
                  sampling_params: SamplingParams, golden_answers: list[str]):
    with VllmRunner(**runner_kwargs) as runner:
        vllm_aclgraph_outputs = runner.model.generate(
            prompts=prompts, sampling_params=sampling_params)
    outputs_gen = []
    for output in vllm_aclgraph_outputs:
        outputs_gen.append(([output.outputs[0].index], output.outputs[0].text))

    output_origin = [([0], answer) for answer in golden_answers]

    check_outputs_equal(
        outputs_0_lst=output_origin,
        outputs_1_lst=outputs_gen,
        name_0="output_origin",
        name_1="outputs_gen",
    )
