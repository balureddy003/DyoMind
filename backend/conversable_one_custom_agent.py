from autogen import ConversableAgent

class ConversableOneCustomAgent(ConversableAgent):
    def __init__(self, name, llm_config, system_message="", description=""):
        super().__init__(
            name=name,
            llm_config=llm_config,
            system_message=system_message,
            description=description
        )