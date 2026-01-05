import pytest

from tests.e2e.singlecard.utils import LLMTestCase, gen_and_valid

PROMPTS_SHORT = [
    "Hello, my name is", "The president of the United States is",
    "The capital of France is", "The future of AI is"
]

# NOTE: Randomly fill the prompt with the requested amount for
# the specified capture shape to prevent accuracy issues caused by padding
PROMPTS_LONG = [
    ('Solve the following math problem step by step.'
     'The last line of your response should be of the form Answer: '
     '$Answer (without quotes) where $Answer is the answer to the problem.\n\n'
     'In triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$'
     'be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$,'
     '$\\angle BDC = 90^\\circ$. Suppose $AD = 1$ and $\\frac{BD}{CD} = \\frac{3}{2}$.'
     'If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$,'
     'where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.'
     ),
    ('Solve the following math problem step by step.'
     'The last line of your response should be of the form Answer: '
     '$Answer (without quotes) where $Answer is the answer to the problem.\n\n'
     'Let $ABCD$ be a unit square in the plane. Points $X$ and $Y$ are chosen'
     'independently and uniformly at random on the perimeter of $ABCD$.'
     'If the expected value of the area of triangle $\\triangle AXY$'
     'can be expressed as $\\frac{m}{n}$, for relatively prime positive'
     'integers $m$ and $n$, compute $m+n$.'),
    ('Solve the following math problem step by step.'
     'The last line of your response should be of the form Answer: '
     '$Answer (without quotes) where $Answer is the answer to the problem.\n\n'
     'Let $a, b, c$ be distinct numbers such that the equations $x^2 + ax + 1 = 0$'
     'and $x^2 + bx + c = 0$ have a common real root, and the equations $x^2 + x + a = 0$'
     'and $x^2 + cx + b = 0$ also have a common real root.'
     'Compute the sum $a + b + c$.')
]

CASE_QWEN_ACLGRAPH = LLMTestCase(
    model="Qwen/Qwen3-0.6B",
    prompts=PROMPTS_SHORT,
    golden_answers=[
        " Lina. I'm a 22-year-old student from China. I'm interested in studying in the US. I'm looking for a job in the",
        ' the same as the president of the United Nations. This is because the president of the United States is the same as the president of the United Nations. The president',
        ' Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of China is Beijing. The capital of Japan is Tokyo. The capital',
        ' not just a technological challenge but a profound transformation of how we live, work, and interact with the world. As we stand at the intersection of artificial intelligence and'
    ],
)

CASE_DS_ACLGRAPH = LLMTestCase(
    model="vllm-ascend/DeepSeek-V2-Lite-W8A8",
    quantization="ascend",
    prompts=PROMPTS_SHORT,
    golden_answers=[
        '\nI am a 20 year old student from the UK. I am currently studying for a degree in English Literature and Creative Writing. I have a passion',
        ' a man who has been in the public eye for decades. He has been a senator, a governor, and a businessman. He has also been married to the',
        ' Paris, which is also the largest city in the country. The city is located on the River Seine and is known for its beautiful architecture, museums, and art',
        ' here.\nThe future of AI is here.\nThe future of AI is here.\nThe future of AI is here.\nThe future of AI is'
    ],
)

CASE_QWEN_FULL_DECODE_ONLY = LLMTestCase(
    model="Qwen/Qwen3-0.6B",
    prompts=PROMPTS_LONG,
    golden_answers=[
        ' \n\nTo solve this problem, we need to use the Law of Sines and Law of Cosines. Let me start by drawing triangle $ABC$ with the',
        " \n\nTo solve this problem, we can use the fact that the expected value of the area of a triangle formed by two random points on a square's perimeter is",
        ' \n\nTo solve this problem, we can use the following approach: Let $ \\alpha $ be the common real root of the two equations. Then, we can'
    ])

CASE_DS_FULL_DECODE_ONLY = LLMTestCase(
    model="vllm-ascend/DeepSeek-V2-Lite-W8A8",
    quantization="ascend",
    prompts=PROMPTS_LONG,
    golden_answers=[
        '\n\nSelect an assignment template',
        '\n\nSelect an assignment template',
        '\n\nSelect an assignment template'
    ])


@pytest.mark.parametrize("cur_case", [CASE_QWEN_ACLGRAPH, CASE_DS_ACLGRAPH])
def test_piecewise_res_consistency(cur_case: LLMTestCase):
    runner_kwargs = {
        "model_name": cur_case.model,
        "max_model_len": 1024,
        "cudagraph_capture_sizes": [1, 2, 4, 8],
        "quantization": cur_case.quantization,
    }
    gen_and_valid(runner_kwargs=runner_kwargs,
                  prompts=cur_case.prompts,
                  sampling_params=cur_case.sampling_params,
                  golden_answers=cur_case.golden_answers)


@pytest.mark.parametrize(
    "cur_case", [CASE_QWEN_FULL_DECODE_ONLY, CASE_DS_FULL_DECODE_ONLY])
def test_full_decode_only_res_consistency(cur_case: LLMTestCase, monkeypatch):
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)
    runner_kwargs = {
        "model_name": cur_case.model,
        "max_model_len": 1024,
        "compilation_config": {
            "cudagraph_capture_sizes": [4, 8, 32, 64],
            "cudagraph_mode": "FULL_DECODE_ONLY"
        },
        "quantization": cur_case.quantization,
    }
    gen_and_valid(runner_kwargs=runner_kwargs,
                  prompts=cur_case.prompts,
                  sampling_params=cur_case.sampling_params,
                  golden_answers=cur_case.golden_answers)


@pytest.mark.parametrize(
    "cur_case", [CASE_QWEN_FULL_DECODE_ONLY, CASE_DS_FULL_DECODE_ONLY])
def test_npugraph_ex_res_consistency(cur_case: LLMTestCase, monkeypatch):
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)
    runner_kwargs = {
        "model_name": cur_case.model,
        "max_model_len": 1024,
        "compilation_config": {
            "cudagraph_capture_sizes": [4, 8, 32, 64],
            "cudagraph_mode": "FULL_DECODE_ONLY"
        },
        "additional_config": {
            "enable_npugraph_ex": True
        },
    }
    gen_and_valid(runner_kwargs=runner_kwargs,
                  prompts=cur_case.prompts,
                  sampling_params=cur_case.sampling_params,
                  golden_answers=cur_case.golden_answers)
