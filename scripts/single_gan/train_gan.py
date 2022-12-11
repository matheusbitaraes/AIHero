from json import load as jload

from src.GEN.service.GENService import GENService

work_dir = 'scripts/single_gan'
with open(f'{work_dir}/config.json') as config_file:
    config = jload(config_file)

# # Script para reduzir dataset
# prefixes = ["X", "Y", "Z"]
# for prefix in prefixes:
#     a = np.load(f'{work_dir}/resources/{prefix}spr_data.npy')
#     aa = a[0:1, :, :, 0:1]
#     b = np.load(f'{work_dir}/resources/{prefix}chord_data.npy')
#     bb = b[0:1, :]
#     np.save(f"{work_dir}/resources/{prefix}spr_data", aa)
#     np.save(f"{work_dir}/resources/{prefix}chord_data", bb)

# realiza treinamento com dados overfittados
gan_service = GENService(config)
gan_service.train_models(should_generate_gif=True)
