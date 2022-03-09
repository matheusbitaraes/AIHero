import os

from src.worker.QueueConsumer import QueueConsumer


class QueueService:
    def __init__(self, config):
        self.melody_path = config["queue"]["melody_path"]
        self.queue_consumer = QueueConsumer(config)

    def add_to_queue(self, melody_request):
        return self.queue_consumer.add_to_queue(melody_request)

    def get_melody_by_id(self, melody_id):
        file = f"{self.melody_path}/{melody_id}.mid"
        print(f"getting melody id {melody_id} on path {self.melody_path}")
        is_file = os.path.isfile(file)
        if not is_file:
            return None
        else:
            print(f"File found at {file}!")
            return file


