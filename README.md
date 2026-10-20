# Sistema de Ponto — POC (DeepFace + Weaviate)

Controle de ponto/acesso por reconhecimento facial usando a câmera do laptop.

- **DeepFace** (`Facenet512` + detector `opencv`) gera um embedding de 512 dimensões do rosto.
- **Weaviate** guarda os embeddings (coleção `Pessoa`, vetores próprios, distância cosine) e as batidas de ponto (coleção `RegistroPonto`).
- No monitoramento, cada rosto vira uma busca `near_vector`; se a distância for menor que o limiar, o acesso é liberado e o ponto é registrado (entrada/saída alternando no dia).

## Setup

```bash
docker compose up -d          # Weaviate em localhost:8080 (gRPC 50051)
uv sync                       # Python 3.11 + deepface, tensorflow, opencv, weaviate-client
```

Na primeira execução a DeepFace baixa os pesos do modelo (~95 MB) para `~/.deepface`, e o macOS
pede permissão de câmera para o terminal (Ajustes do Sistema › Privacidade e Segurança › Câmera).

Se as portas 8080/50051 já estiverem em uso, troque as duas pelas mesmas variáveis
(valem para o `docker compose` e para o `ponto.py`):

```bash
export WEAVIATE_PORT=8090 WEAVIATE_GRPC_PORT=50061
```

## Uso

```bash
# Cadastro: ESPAÇO captura uma amostra, "a" liga o modo automático (1/s), "q" sai
uv run python ponto.py cadastrar --nome "Duda" --amostras 5

# Monitoramento / controle de acesso ("q" sai)
uv run python ponto.py monitorar                 # limiar 0.30, cooldown 60s
uv run python ponto.py monitorar --limiar 0.25   # mais restritivo

uv run python ponto.py listar                    # pessoas e nº de amostras
uv run python ponto.py registros --dia 2026-10-20
uv run python ponto.py remover --nome "Duda"
uv run python ponto.py reset                     # apaga tudo
```

Opção global `--camera N` escolhe outra câmera (ex.: `uv run python ponto.py --camera 1 monitorar`).

## Calibração

A tela mostra a distância (`dist=`) do vizinho mais próximo. Com Facenet512 + cosine, a mesma pessoa
costuma ficar abaixo de ~0.30. Se pessoas cadastradas forem negadas, cadastre mais amostras
(ângulos e iluminações diferentes) ou suba o limiar; se estranhos passarem, baixe o limiar.

## Limitações (é uma POC)

- Sem *liveness detection*: uma foto do rosto na frente da câmera pode liberar o acesso.
- Sem autenticação no Weaviate (acesso anônimo, só local).
