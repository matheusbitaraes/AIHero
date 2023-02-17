from json import load as jload
from src.GEN.service.GENService import GENService

with open('src/config.json') as config_file:
    config = jload(config_file)

gen_service_conv = GENService(config)
gen_service_conv.train_models(should_generate_gif=True)

#
# with open('src/lstm_config.json') as config_file:
#     config_lstm = jload(config_file)
# gen_service_lstm = GENService(config_lstm)
# gen_service_lstm.train_models(should_generate_gif=False)


