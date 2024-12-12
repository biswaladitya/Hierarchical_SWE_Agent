import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from sweagent.agent.agents import AgentArguments, Agent
from sweagent.agent.models import APIStats
from sweagent.environment.swe_env import SWEEnv
from sweagent.types import TrajectoryStep
from sweagent.utils.config import keys_config

org_agent_prompt = """
You are an agent to organize a team of agents to solve a task within a project. 
The task is to solve the following issue within a code base {issue_description}.
You can choose from the following agents: {available_agents}
Within every message iterate over the following steps, without repeating the output of a tool call: 
Following the Chain of Thought prompting method you should decide on the task that needs to be done.
Then you can choose an agent to run the task.
Start by writing down your considerations about the status of the project and the task that needs to be done.
CONSIDERATIONS: ...
NEXT STEPS...
Then write down the next specific sub-task that needs to be done and choose an agent to run the task using the provided function call. Specify a relevant definition of done.
This subtask is only one step in the process of solving the issue. After the called agent will return the result, you can decide on the next subtask and continue to call agents (the same or others) until the issue is solved.
If you think the most recent interaction appropriately solves the task, you can end the conversation by writing DONE. Otherwise please always provide a new subtask to be done and call a relevant agent to run it.
"""
# TODO Optimize this prompt
# TODO which kind of agents do we need that the org agent can work with?

# TODO improve the edit and other used commands to be more agent friendly

# TODO Experiment with the organization agent. Find out if multiple sub_agents are helpful or if one agent can do the job.

class OrgAgent:

    def __init__(self, name, args: AgentArguments):
        self.name = name
        self.args = args
        self.available_agents = {
            "coder": {
                "agent": Agent('coder_sub_agent', self.args),
                "description": "Agent to code the solution with environment access",
            },
            "tester": {
                "agent": Agent('tester_sub_agent', self.args),
                "description": "Agent focussed on testing the solution with full environment access",
            },
            "reviewer": {
                "agent": Agent('reviewer_sub_agent', self.args),
                "description": "Agent to review the code and provide feedback",
            },
            'planner': {
                "agent": Agent('planner_sub_agent', self.args),
                "description": "Agent to plan the solution and create a roadmap",
            },
        }
        self.prompt = None

    def run(
        self,
        setup_args: dict[str, Any],
        env: SWEEnv,
        observation: str = None,
        traj_dir: Path = None,
        return_type: str = "info_trajectory",
        init_model_stats: APIStats = None,
    ):
        """
        Run an organization agent to organize a team of agents to solve a task within a project.
        For this to accurately work, set the --config_file arg to the config/default_org_agent.yaml file or use the preconfigured run_config in intelliJ

        """
        self.prompt = org_agent_prompt.format(
            available_agents=", ".join(
                [agent_name + ": " + agent["description"] for agent_name, agent in self.available_agents.items()]
            ),
            issue_description=setup_args["issue"],
        )
        client = OpenAI(
            api_key=keys_config["OPENAI_API_KEY"],
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "call_agents",
                    "description": "This function let's you call agents to solve a task."
                    "Available agents are:"
                    + ", ".join(
                        [
                            agent_name + ": " + agent["description"]
                            for agent_name, agent in self.available_agents.items()
                        ]
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent": {
                                "type": "string",
                                "description": "The agent to call",
                                "enum": [agent_name for agent_name in self.available_agents.keys()],
                            },
                            "task": {
                                "type": "string",
                                "description": "An extensive description of the task to be done and the scope that the agent should fulfill.",
                            },
                            "current_status": {
                                "type": "string",
                                "description": "An extensive description of the current status of the issue solving process.",
                            },
                            "definition_of_done": {
                                "type": "string",
                                "description": "A clear description of the expected output of the task, that the agent should fulfill and can use to determine when the task is done.",
                            },
                        },
                    },
                },
            }
        ]
        messages = [
            {
                "role": "system",
                "content": self.prompt,
            }
        ]

        combined_info = {
            "model_stats": init_model_stats or APIStats(),
        }

        combined_trajectory = []
        while True:
            chat = client.chat.completions.create(
                messages=messages,
                model="gpt-4o-mini",
                tools=tools,
            )
            choice = chat.choices[0]
            messages.append(choice.message)
            print("ORGANIZATION AGENT", choice.message.content)
            combined_info["model_stats"] += APIStats(
                api_calls = 1 ,
                tokens_sent= chat.usage.prompt_tokens ,
                tokens_received = chat.usage.completion_tokens,
                instance_cost = (chat.usage.prompt_tokens * 0.00000015 + chat.usage.completion_tokens * 0.0000006),
                total_cost= (chat.usage.prompt_tokens * 0.00000015 + chat.usage.completion_tokens * 0.0000006),
            )
            if choice.finish_reason == "tool_calls":
                tool_calls = choice.message.tool_calls
                for tool_call in tool_calls:
                    arguments = json.loads(tool_call.function.arguments)
                    agent_to_call = self.available_agents[arguments["agent"]]

                    setup_args["agent_description"] = agent_to_call["description"]
                    setup_args["current_status"] = arguments["current_status"]
                    setup_args["task"] = arguments["task"]
                    setup_args["definition_of_done"] = arguments["definition_of_done"]
                    combined_trajectory.append(TrajectoryStep(
                        {
                            "action": str(arguments),
                            "observation": choice.message.content,
                            "response": None,
                            "state": None,
                            "thought": None,
                            "execution_time": 0,
                        },
                    ))
                    recent_info, recent_trajectory, summary = agent_to_call["agent"].run(
                        setup_args,
                        env,
                        observation=observation,
                        traj_dir=traj_dir,
                        return_type="summary",
                        init_model_stats=combined_info['model_stats'],
                    )
                    combined_info["model_stats"] = APIStats(**recent_info["model_stats"])
                    print("TOOL AGENT", summary)
                    # TODO handle trajectory. Is that necessary? What is it, how to combine, what for? Kinda done by the following line
                    combined_trajectory.extend(recent_trajectory)
                    messages.append({"role": "tool", "content": summary, "tool_call_id": tool_call.id})
            if chat.choices[-1].finish_reason == "stop":
                if "DONE" in chat.choices[-1].message.content:
                    # TODO add a check if the task is done and what then? Calling the env submit function?
                    # TODO implement a collective AgentInfo object that collects the info from all agents: Every Agent tool call will return that.
                    # We have to look into how to combine those. Could be helpful for stats or something
                    # Info contains stuff like how many tokens have been called, estimated price, etc.
                    break

        return combined_info, combined_trajectory
