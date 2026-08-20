class BaseAgent:
    
    

    def __init__(self, agent_id, name):

        self.agent_id = agent_id
        self.name = name
        self.status = "Available"

        # Boîte de réception
        self.inbox = []

    def receive_message(self, message):
        """
        Reçoit un message et le stocke.
        """
        self.inbox.append(message)

    def show_messages(self):
        """
        Affiche les messages reçus.
        """
        print(f"\nMessages of {self.name}")

        for msg in self.inbox:
            print(msg)

    def __str__(self):

        return f"{self.name} ({self.agent_id})"