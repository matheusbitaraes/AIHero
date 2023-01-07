import queue
import threading
import uuid

from src.model.ApiModels import MelodyRequest
from src.service.AIHeroService import AIHeroService


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
                                       melody_specs=melody_request_input.melody_specs)
        self.q.put(melody_request)
        print(f"Adding melody request {melody_request.id} into queue ")
        return melody_request.id

    def worker(self):
        while True:
            item = self.q.get()
            melody_id = item.id
            source = item.source
            harmony_specs = item.melody_specs.harmony_specs
            evolutionary_specs = item.melody_specs.evolutionary_specs
            harmony_file = "src/resources/blues_base.mid"  # todo criar algum tipo de mapa para resolver isso
            print(f'Working on  melody {melody_id} \n {item}')
            if source == "evo":
                result = self.ai_hero_service.generate_compositions(harmony_specs,
                                                                    evolutionary_specs=evolutionary_specs,
                                                                    melody_id=melody_id)
                result.append_track_and_export_as_midi(file_name=f"{self.melody_path}/{melody_id}",
                                                       midi_file=harmony_file)
            if source == "gan":
                result = self.ai_hero_service.generate_GEN_compositions(harmony_specs, melody_id=melody_id)
                result.append_track_and_export_as_midi(file_name=f"{self.melody_path}/{melody_id}",
                                                       midi_file=harmony_file)
            if source == "train":
                result = self.ai_hero_service.generate_compositions_with_train_data(harmony_specs,
                                                                                    melody_id=melody_id)
                result.append_track_and_export_as_midi(file_name=f"{self.melody_path}/{melody_id}",
                                                       midi_file=harmony_file)
            print(f'Finished {melody_id}')
            self.q.task_done()
