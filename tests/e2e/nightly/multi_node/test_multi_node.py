import pytest

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.nightly.multi_node.scripts.multi_node_config import \
    MultiNodeConfig
from tools.aisbench import run_aisbench_cases


@pytest.mark.asyncio
async def test_multi_node() -> None:
    # ========= Load config =========
    config = MultiNodeConfig.from_yaml()

    envs = config.envs
    model = config.model

    server_host = config.master_ip
    server_port = config.server_port
    proxy_port = config.proxy_port

    nodes_info = config.nodes_info
    disaggregated_prefill = config.disaggregated_prefill

    perf_cmd = config.perf_cmd
    acc_cmd = config.acc_cmd

    proxy_script = envs.get(
        "DISAGGREGATED_PREFILL_PROXY_SCRIPT",
        "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py",
    )

    with config.launch_server_proxy(proxy_script):
        with RemoteOpenAIServer(
                model=model,
                vllm_serve_args=config.server_cmd,
                server_host=server_host,
                server_port=server_port,
                proxy_port=proxy_port,
                env_dict=envs,
                auto_port=False,
                disaggregated_prefill=disaggregated_prefill,
                nodes_info=nodes_info,
                max_wait_seconds=1200,
        ) as remote_server:

            if config.is_master:
                _run_master_node_tests(
                    config=config,
                    model=model,
                    server_port=server_port,
                    proxy_port=proxy_port,
                    acc_cmd=acc_cmd,
                    perf_cmd=perf_cmd,
                )
            else:
                _wait_as_worker_node(
                    remote_server=remote_server,
                    master_ip=config.master_ip,
                    server_port=server_port,
                )


def _run_master_node_tests(
    *,
    config: MultiNodeConfig,
    model: str,
    server_port: int,
    proxy_port: int,
    acc_cmd: str,
    perf_cmd: str,
) -> None:
    port = proxy_port if config.disaggregated_prefill else server_port

    aisbench_cases = [acc_cmd, perf_cmd]

    run_aisbench_cases(
        model=model,
        port=port,
        aisbench_cases=aisbench_cases,
        host_ip=config.master_ip,
    )


def _wait_as_worker_node(
    *,
    remote_server: RemoteOpenAIServer,
    master_ip: str,
    server_port: int,
) -> None:
    master_health_url = f"http://{master_ip}:{server_port}/health"
    remote_server.hang_until_terminated(master_health_url)
