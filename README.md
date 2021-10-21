# AIHero
AI for music improvisation

TODO: insert image of the flow here

Tarefas:
 - [ ] Definição embasada dos modelos gerador e discriminador da GAN.
 - [ ] Colocar plots que mostram evolução das perdas no gerador e discriminador
 - [X] Reflexão sobre como adequar melodias aos acordes: fazer piano_rolls padronizados pelos acordes
 - [ ] criar testes para validar os scaled_piano_rolls
 - [X] Adequar dados de teste atual e treinamento para usar scaled_piano_roll
 - [X] Geração de mais amostras de teste.
 - [X] Implementação da execução da melodia e montagem com a base.
 - [ ] Colocar algoritmo evolucionário no processo.
 - [ ] Adicionar velocidades aos piano_rolls e criar mais esse conceito?



TODO:
- fazer algoritmo genético convergir melhor.
- fazer os intervalos virarem duraçoes aleatorias das notas! tipo, se tem só 1 nota no tempo, ela pode assumir a duraçao da fusa ate a seminima.
- Desenhar arquitetura.
- Fazer o algoritmo se comportar de uma forma "ao vivo", onde os parametros vão mudando e as melodias também.

Para o fitness, a ideia de agora é colocar uma rede neural para avaliar a proximidade da melodia gerada com uma melodia real.
para a inicialização da população, pode ser usada uma outra rede neural para gerar.
e aí também terão os objetivos inseridos pelo usuario (variedade de notas, quantidade de notas, pitch...e é isso que o algoritmo genético vai tentar atacar)

## Architecture

## The genetic algorithm

## The graphical interface


## Next steps:



