import queue
import threading
import uuid

from service.AIHeroService import AIHeroService

from model.ApiModels import MelodyRequest


class QueueConsumer:
    def __init__(self, config):
        self.melody_path = config["queue"]["melody_path"]
        self.q = queue.Queue()
        self.ai_hero_service = AIHeroService(config)
        # self.file_manager = boto3.resource("s3")
        # Print out bucket names
        # for bucket in self.file_manager.buckets.all():
        #     print(bucket.name)
        threading.Thread(target=self.worker, daemon=True).start()

    def add_to_queue(self, melody_request_input):
        melody_request = MelodyRequest(id=uuid.uuid4(),
                                       source=melody_request_input.source,
                                       melody_specs_list=melody_request_input.melody_specs_list)
        self.q.put(melody_request)
        print(f"Adding melody request {melody_request.id} into queue ")
        return melody_request.id

    def worker(self):
        while True:
            item = self.q.get()
            melody_id = item.id
            source = item.source
            melody_specs_list = item.melody_specs_list
            print(f'Working on  melody {melody_id} \n {item}')
            if source == "evo":
                result = self.ai_hero_service.generate_compositions(melody_specs_list,
                                                                    melody_id=melody_id)
                result.export_as_midi(file_name=f"{self.melody_path}/{melody_id}")
            if source == "gan":
                result = self.ai_hero_service.generate_GAN_compositions(melody_specs_list,
                                                                        melody_id=melody_id)
                result.export_as_midi(file_name=f"{self.melody_path}/{melody_id}")
            if source == "train":
                result = self.ai_hero_service.generate_compositions_with_train_data(melody_specs_list,
                                                                                    melody_id=melody_id)
                result.export_as_midi(file_name=f"{self.melody_path}/{melody_id}")
            print(f'Finished {melody_id}')
            self.q.task_done()
