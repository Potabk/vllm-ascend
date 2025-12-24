import logging
import os
import subprocess
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import regex as re
import yaml

from tests.e2e.nightly.multi_node.scripts.utils import (get_all_ipv4,
                                                        get_avaliable_port,
                                                        get_cluster_ips,
                                                        get_net_interface,
                                                        setup_logger)

# ----------------------------------------------------------------------
# logging / constants
# ----------------------------------------------------------------------

setup_logger()
logger = logging.getLogger(__name__)

CONFIG_BASE_PATH = "tests/e2e/nightly/multi_node/config/"
DEFAULT_SERVER_PORT = 8080

# ----------------------------------------------------------------------
# Node description
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class NodeInfo:
    # Global index of the node in the cluster
    index: int
    ip: str
    server_cmd: str
    headless: bool
    server_port: int

    def __str__(self) -> str:
        return ("NodeInfo:\n"
                f"  index={self.index}\n"
                f"  ip={self.ip}\n"
                f"  headless={self.headless}\n"
                f"  server_port={self.server_port}")

    @property
    def is_master(self) -> bool:
        return self.index == 0


# ----------------------------------------------------------------------
# Roles & topology
# ----------------------------------------------------------------------


class NodeRole(Enum):
    NO_PD = auto()
    PREFILLER = auto()
    DECODER = auto()


@dataclass(frozen=True)
class DisaggregatedPrefillConfig:
    prefiller_indices: list[int]
    decoder_indices: list[int]

    @classmethod
    def from_dict(cls, cfg: dict) -> "DisaggregatedPrefillConfig":
        return cls(
            prefiller_indices=cfg["prefiller_host_index"],
            decoder_indices=cfg["decoder_host_index"],
        )

    def role_of(self, index: int) -> NodeRole:
        if index in self.prefiller_indices:
            return NodeRole.PREFILLER
        if index in self.decoder_indices:
            return NodeRole.DECODER
        raise ValueError(
            f"Index {index} not in disaggregated prefill topology")

    @property
    def decoder_master_index(self) -> int:
        return self.decoder_indices[0]


@dataclass(frozen=True)
class ClusterTopology:
    nodes: list[NodeInfo]
    disaggregated_pd: Optional[DisaggregatedPrefillConfig] = None

    @property
    def master_ip(self) -> str:
        return self.nodes[0].ip

    def role_of(self, index: int) -> NodeRole:
        if not self.disaggregated_pd:
            return NodeRole.NO_PD
        return self.disaggregated_pd.role_of(index)

    def master_ip_for(self, index: int) -> str:
        role = self.role_of(index)
        if role == NodeRole.NO_PD or role == NodeRole.PREFILLER:
            return self.master_ip

        if role == NodeRole.DECODER:
            master_index = self.disaggregated_pd.decoder_master_index
            return self.nodes[master_index].ip

        raise RuntimeError("Invalid node role")


@dataclass
class MultiNodeRuntimeContext:
    topology: ClusterTopology

    def resolve_index(self) -> int:
        worker_index = os.environ.get("LWS_WORKER_INDEX")
        if worker_index is not None:
            return int(worker_index)

        cluster_ips = [n.ip for n in self.topology.nodes]
        for ip in get_all_ipv4():
            if ip in cluster_ips:
                return cluster_ips.index(ip)

        raise RuntimeError("Failed to resolve current node index.\n"
                           f"Local IPs: {get_all_ipv4()}\n"
                           f"Cluster IPs: {cluster_ips}")

    @property
    def index(self) -> int:
        return self.resolve_index()

    @property
    def node(self) -> NodeInfo:
        return self.topology.nodes[self.index]

    @property
    def ip(self) -> str:
        return self.node.ip

    @property
    def nic(self) -> str:
        return get_net_interface(self.ip)

    @property
    def role(self) -> NodeRole:
        return self.topology.role_of(self.index)

    @property
    def master_ip(self) -> str:
        return self.topology.master_ip_for(self.index)


# ----------------------------------------------------------------------
# Main config object
# ----------------------------------------------------------------------


class MultiNodeConfig:
    """
    Control-plane configuration for multi-node E2E tests.
    """

    def __init__(
        self,
        *,
        model: str,
        test_name: str,
        topology: ClusterTopology,
        npu_per_node: int,
        server_port: int,
        envs: dict,
        perf_cmd: Optional[str],
        acc_cmd: Optional[str],
    ):
        self.model = model
        self.test_name = test_name
        self.topology = topology
        self.npu_per_node = npu_per_node
        self.server_port = server_port

        self.proxy_port = get_avaliable_port()
        self.envs = dict(envs)
        self.perf_cmd = perf_cmd
        self.acc_cmd = acc_cmd

        self.runtime = MultiNodeRuntimeContext(topology)
        self._init_envs()

    def _init_envs(self) -> None:
        r = self.runtime
        self.envs.update({
            "LOCAL_IP": r.ip,
            "NIC_NAME": r.nic,
            "MASTER_IP": r.master_ip,
            "HCCL_IF_IP": r.ip,
            "TP_SOCKET_IFNAME": r.nic,
            "GLOO_SOCKET_IFNAME": r.nic,
            "HCCL_SOCKET_IFNAME": r.nic,
        })

        # ensure string envs
        self.envs = {k: str(v) for k, v in self.envs.items()}

    @property
    def is_master(self) -> bool:
        return self.runtime.index == 0

    @property
    def server_cmd(self) -> str:
        return self._expand_env_vars(
            self.runtime.node.server_cmd,
            self.envs,
        )

    @property
    def world_size(self) -> int:
        return len(self.topology.nodes) * self.npu_per_node

    def launch_server_proxy(self, proxy_script: str):
        return ProxyContext(self, proxy_script)

    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None) -> "MultiNodeConfig":
        yaml_path = yaml_path or os.getenv("CONFIG_YAML_PATH",
                                           "DeepSeek-V3.yaml")
        yaml_path = os.path.join(CONFIG_BASE_PATH, yaml_path)

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        topology = cls._parse_topology(cfg)
        benchmarks = cfg.get("benchmarks") or {}

        return cls(
            model=cfg.get("model", "default_model"),
            test_name=cfg.get("test_name", "default_test"),
            topology=topology,
            npu_per_node=cfg.get("npu_per_node", 16),
            server_port=cfg.get("server_port", DEFAULT_SERVER_PORT),
            envs=cfg.get("env_common", {}),
            perf_cmd=benchmarks.get("perf"),
            acc_cmd=benchmarks.get("acc"),
        )

    @staticmethod
    def _parse_topology(cfg: dict) -> ClusterTopology:
        num_nodes = cfg["num_nodes"]
        deployments = cfg["deployment"]

        assert len(deployments) == num_nodes

        cluster_ips = cfg.get("cluster_hosts") or get_cluster_ips(num_nodes)

        nodes: list[NodeInfo] = []
        for i, dep in enumerate(deployments):
            cmd = dep.get("server_cmd", "")
            nodes.append(
                NodeInfo(
                    index=i,
                    ip=cluster_ips[i],
                    server_cmd=cmd,
                    headless="--headless" in cmd,
                    server_port=cfg.get("server_port", DEFAULT_SERVER_PORT),
                ))

        pd_cfg = cfg.get("disaggregated_prefill")
        disaggregated_pd = (DisaggregatedPrefillConfig.from_dict(pd_cfg)
                            if pd_cfg else None)

        return ClusterTopology(nodes=nodes, disaggregated_pd=disaggregated_pd)

    @staticmethod
    def _expand_env_vars(cmd: str, env: dict) -> str:
        pattern = re.compile(r"\$(\w+)|\$\{(\w+)\}")

        def repl(match):
            key = match.group(1) or match.group(2)
            return env.get(key, match.group(0))

        return pattern.sub(repl, str(cmd))


class ProxyContext:

    def __init__(self, cfg: MultiNodeConfig, script: str):
        self.cfg = cfg
        self.script = script
        self.process: Optional[subprocess.Popen] = None

    def __enter__(self):
        rt = self.cfg.runtime
        topo = self.cfg.topology

        if rt.role != NodeRole.MASTER or topo.disaggregated_pd is None:
            logger.info("Proxy not required on this node.")
            return self

        self.process = self._launch_proxy(rt, topo)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.process:
            logger.info("Stopping proxy server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _launch_proxy(
        self,
        rt: MultiNodeRuntimeContext,
        topo: ClusterTopology,
    ) -> subprocess.Popen:
        pre = topo.disaggregated_pd
        assert pre is not None

        ips = [n.ip for n in topo.nodes]

        cmd = [
            "python",
            self.script,
            "--host",
            rt.ip,
            "--port",
            str(self.cfg.proxy_port),
            "--prefiller-hosts",
            *(ips[i] for i in pre.prefiller_indices),
            "--prefiller-ports",
            *(str(self.cfg.server_port), ) * len(pre.prefiller_indices),
            "--decoder-hosts",
            *(ips[i] for i in pre.decoder_indices),
            "--decoder-ports",
            *(str(self.cfg.server_port), ) * len(pre.decoder_indices),
        ]

        logger.info("Launching proxy: %s", " ".join(cmd))
        env = {**os.environ, **self.cfg.envs}
        return subprocess.Popen(cmd, env=env)


if __name__ == '__main__':
    config = MultiNodeConfig.from_yaml()
    logger.info("Loaded multi-node config for test '%s'", config.test_name)
    logger.info("Cluster topology:")
    for node in config.topology.nodes:
        logger.info("%s", node)
