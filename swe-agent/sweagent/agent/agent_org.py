from pathlib import Path
from typing import Any

from sweagent.agent.agents import AgentArguments, Agent
from sweagent.agent.models import APIStats
from sweagent.environment.swe_env import SWEEnv


class OrgAgent():

    def __init__(self, name, args: AgentArguments):
        self.agent = Agent(name, args)

    # make every method in Agent class available in OrgAgent class
    def __getattr__(self, name):
        return getattr(self.agent, name)

    def __setattr__(self, name, value):
        if name in self.__dict__.keys():
            self.__dict__[name] = value
        else:
            setattr(self.agent, name, value)

    def run(
        self,
        setup_args: dict[str, Any],
        env: SWEEnv,
        observation: str | None = None,
        traj_dir: Path | None = None,
        return_type: str = "info_trajectory",
        init_model_stats: APIStats | None = None,
    ):
        return self.agent.run(
            setup_args,
            env,
            observation=observation,
            traj_dir=traj_dir,
            return_type=return_type,
            init_model_stats=init_model_stats,
        )