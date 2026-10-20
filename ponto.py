"""POC de controle de ponto com reconhecimento facial.

Usa a câmera do laptop, gera embeddings faciais com DeepFace e usa o Weaviate
como banco vetorial para identificar a pessoa e registrar entrada/saída.

Uso:
    python ponto.py cadastrar --nome "Fulano"
    python ponto.py monitorar
    python ponto.py listar
    python ponto.py registros [--dia 2026-10-15]
    python ponto.py remover --nome "Fulano"
    python ponto.py reset
"""

import argparse
import os
import sys
import time
import uuid
from datetime import datetime, timezone

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization, VectorDistances
from weaviate.classes.query import Filter, MetadataQuery, Sort

WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8090"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50061"))

MODELO = "Facenet512"
DETECTOR = "opencv"
COL_PESSOA = "Pessoa"
COL_REGISTRO = "RegistroPonto"

VERDE = (0, 200, 0)
VERMELHO = (0, 0, 220)
AMARELO = (0, 200, 220)
BRANCO = (255, 255, 255)


# --------------------------------------------------------------------------- #
# Weaviate
# --------------------------------------------------------------------------- #
def conectar():
    client = weaviate.connect_to_local(port=WEAVIATE_PORT, grpc_port=WEAVIATE_GRPC_PORT)
    garantir_colecoes(client)
    return client


def garantir_colecoes(client):
    # Propriedades de texto usam tokenização FIELD: os filtros por igualdade
    # (nome, pessoa_id, dia) precisam casar o valor inteiro, não palavras soltas.
    vetores = Configure.Vectors.self_provided(
        vector_index_config=Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)
    )
    if not client.collections.exists(COL_PESSOA):
        client.collections.create(
            COL_PESSOA,
            vector_config=vetores,
            properties=[
                Property(name="nome", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="pessoa_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="cadastrado_em", data_type=DataType.DATE),
            ],
        )
    if not client.collections.exists(COL_REGISTRO):
        client.collections.create(
            COL_REGISTRO,
            vector_config=Configure.Vectors.self_provided(),
            properties=[
                Property(name="pessoa_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="nome", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="tipo", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),  # entrada | saida
                Property(name="dia", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),  # YYYY-MM-DD (hora local)
                Property(name="timestamp", data_type=DataType.DATE),
                Property(name="distancia", data_type=DataType.NUMBER),
            ],
        )


def buscar_pessoa(client, embedding, limiar):
    """Retorna (nome, pessoa_id, distancia) do vizinho mais próximo, ou None."""
    res = client.collections.use(COL_PESSOA).query.near_vector(
        near_vector=embedding,
        limit=1,
        return_metadata=MetadataQuery(distance=True),
    )
    if not res.objects:
        return None
    obj = res.objects[0]
    dist = obj.metadata.distance
    return obj.properties["nome"], obj.properties["pessoa_id"], dist, dist <= limiar


def ultimo_registro_do_dia(client, pessoa_id, dia):
    res = client.collections.use(COL_REGISTRO).query.fetch_objects(
        filters=Filter.by_property("pessoa_id").equal(pessoa_id)
        & Filter.by_property("dia").equal(dia),
        sort=Sort.by_property("timestamp", ascending=False),
        limit=1,
    )
    return res.objects[0].properties if res.objects else None


def bater_ponto(client, nome, pessoa_id, distancia):
    agora = datetime.now().astimezone()
    dia = agora.strftime("%Y-%m-%d")
    ultimo = ultimo_registro_do_dia(client, pessoa_id, dia)
    tipo = "saida" if ultimo and ultimo["tipo"] == "entrada" else "entrada"
    client.collections.use(COL_REGISTRO).data.insert(
        {
            "pessoa_id": pessoa_id,
            "nome": nome,
            "tipo": tipo,
            "dia": dia,
            "timestamp": agora,
            "distancia": float(distancia),
        }
    )
    return tipo, agora


# --------------------------------------------------------------------------- #
# DeepFace / câmera
# --------------------------------------------------------------------------- #
def carregar_modelo():
    print(f"Carregando modelo {MODELO} (a primeira vez baixa os pesos)...")
    from deepface import DeepFace

    DeepFace.build_model(MODELO)
    return DeepFace


def extrair_rostos(DeepFace, frame):
    """Lista de dicts {embedding, facial_area, face_confidence}; vazia se não achar rosto."""
    try:
        rostos = DeepFace.represent(
            img_path=frame,
            model_name=MODELO,
            detector_backend=DETECTOR,
            enforce_detection=True,
            align=True,
        )
    except ValueError:  # nenhum rosto detectado
        return []
    return [r for r in rostos if r.get("face_confidence", 1) > 0.5]


def maior_rosto(rostos):
    return max(rostos, key=lambda r: r["facial_area"]["w"] * r["facial_area"]["h"])


def abrir_camera(indice):
    cam = cv2.VideoCapture(indice)
    if not cam.isOpened():
        sys.exit(
            "Não consegui abrir a câmera. No macOS, dê permissão de câmera ao "
            "terminal em Ajustes do Sistema > Privacidade e Segurança > Câmera."
        )
    return cam


def desenhar(frame, area, cor, *linhas):
    x, y, w, h = area["x"], area["y"], area["w"], area["h"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
    for i, texto in enumerate(linhas):
        pos = (x, max(25, y - 10 - 28 * (len(linhas) - 1 - i)))
        cv2.putText(frame, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)


def rodape(frame, texto):
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 35), (frame.shape[1], h), (0, 0, 0), -1)
    cv2.putText(frame, texto, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BRANCO, 1)


# --------------------------------------------------------------------------- #
# Comandos
# --------------------------------------------------------------------------- #
def cmd_cadastrar(args):
    client = conectar()
    try:
        pessoas = client.collections.use(COL_PESSOA)
        existente = pessoas.query.fetch_objects(
            filters=Filter.by_property("nome").equal(args.nome), limit=1
        )
        if existente.objects:
            pessoa_id = existente.objects[0].properties["pessoa_id"]
            print(f"'{args.nome}' já cadastrado; adicionando novas amostras.")
        else:
            pessoa_id = str(uuid.uuid4())

        DeepFace = carregar_modelo()
        cam = abrir_camera(args.camera)
        capturadas = 0
        ultimo_auto = 0.0
        print("ESPAÇO = capturar amostra | a = modo automático | q = sair")
        auto = False
        try:
            while capturadas < args.amostras:
                ok, frame = cam.read()
                if not ok:
                    continue
                exibir = frame.copy()
                rostos = extrair_rostos(DeepFace, frame)
                tecla = cv2.waitKey(1) & 0xFF

                if len(rostos) == 1:
                    desenhar(exibir, rostos[0]["facial_area"], VERDE, "rosto OK")
                    quer_capturar = tecla == ord(" ") or (auto and time.time() - ultimo_auto > 1.0)
                    if quer_capturar:
                        pessoas.data.insert(
                            properties={
                                "nome": args.nome,
                                "pessoa_id": pessoa_id,
                                "cadastrado_em": datetime.now(timezone.utc),
                            },
                            vector=rostos[0]["embedding"],
                        )
                        capturadas += 1
                        ultimo_auto = time.time()
                        print(f"  amostra {capturadas}/{args.amostras} salva")
                elif len(rostos) > 1:
                    for r in rostos:
                        desenhar(exibir, r["facial_area"], AMARELO, "apenas 1 rosto")

                if tecla == ord("a"):
                    auto = not auto
                if tecla == ord("q"):
                    break

                modo = "AUTO" if auto else "manual"
                rodape(
                    exibir,
                    f"Cadastro: {args.nome} | {capturadas}/{args.amostras} | {modo} | "
                    "ESPACO captura, a auto, q sai",
                )
                cv2.imshow("Cadastro - Sistema de Ponto", exibir)
        finally:
            cam.release()
            cv2.destroyAllWindows()
        print(f"Cadastro de '{args.nome}' concluído com {capturadas} amostra(s).")
    finally:
        client.close()


def cmd_monitorar(args):
    client = conectar()
    try:
        if len(client.collections.use(COL_PESSOA)) == 0:
            sys.exit("Nenhuma pessoa cadastrada. Rode: python ponto.py cadastrar --nome ...")

        DeepFace = carregar_modelo()
        cam = abrir_camera(args.camera)
        ultimo_ponto = {}  # pessoa_id -> epoch do último registro (cooldown)
        mensagem, mensagem_ate = "", 0.0
        ultima_analise = 0.0
        resultado = None  # (area, cor, linhas) do último reconhecimento
        print(f"Monitorando (limiar={args.limiar}, cooldown={args.cooldown}s). q = sair")
        try:
            while True:
                ok, frame = cam.read()
                if not ok:
                    continue

                if time.time() - ultima_analise >= args.intervalo:
                    ultima_analise = time.time()
                    rostos = extrair_rostos(DeepFace, frame)
                    resultado = None
                    if rostos:
                        rosto = maior_rosto(rostos)
                        achado = buscar_pessoa(client, rosto["embedding"], args.limiar)
                        if achado and achado[3]:
                            nome, pessoa_id, dist, _ = achado
                            linhas = [f"ACESSO LIBERADO: {nome}", f"dist={dist:.3f}"]
                            resultado = (rosto["facial_area"], VERDE, linhas)
                            agora = time.time()
                            if agora - ultimo_ponto.get(pessoa_id, 0) >= args.cooldown:
                                tipo, quando = bater_ponto(client, nome, pessoa_id, dist)
                                ultimo_ponto[pessoa_id] = agora
                                mensagem = f"Ponto registrado: {nome} - {tipo.upper()} {quando:%H:%M:%S}"
                                mensagem_ate = agora + 4
                                print(mensagem)
                        else:
                            dist = f"dist={achado[2]:.3f}" if achado else "sem cadastro"
                            linhas = ["ACESSO NEGADO", dist]
                            resultado = (rosto["facial_area"], VERMELHO, linhas)

                if resultado:
                    desenhar(frame, resultado[0], resultado[1], *resultado[2])
                rodape(frame, mensagem if time.time() < mensagem_ate else "Olhe para a camera | q sai")
                cv2.imshow("Controle de Ponto", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cam.release()
            cv2.destroyAllWindows()
    finally:
        client.close()


def cmd_listar(args):
    client = conectar()
    try:
        contagem = {}
        for obj in client.collections.use(COL_PESSOA).iterator():
            contagem[obj.properties["nome"]] = contagem.get(obj.properties["nome"], 0) + 1
        if not contagem:
            print("Nenhuma pessoa cadastrada.")
        for nome, n in sorted(contagem.items()):
            print(f"{nome:30s} {n} amostra(s)")
    finally:
        client.close()


def cmd_registros(args):
    client = conectar()
    try:
        filtro = Filter.by_property("dia").equal(args.dia) if args.dia else None
        res = client.collections.use(COL_REGISTRO).query.fetch_objects(
            filters=filtro,
            sort=Sort.by_property("timestamp", ascending=True),
            limit=args.limite,
        )
        if not res.objects:
            print("Nenhum registro.")
        for obj in res.objects:
            p = obj.properties
            ts = p["timestamp"].astimezone()
            print(f"{ts:%Y-%m-%d %H:%M:%S}  {p['tipo']:8s} {p['nome']:30s} dist={p['distancia']:.3f}")
    finally:
        client.close()


def cmd_remover(args):
    client = conectar()
    try:
        res = client.collections.use(COL_PESSOA).data.delete_many(
            where=Filter.by_property("nome").equal(args.nome)
        )
        print(f"{res.successful} amostra(s) de '{args.nome}' removida(s).")
    finally:
        client.close()


def cmd_reset(args):
    client = weaviate.connect_to_local(port=WEAVIATE_PORT, grpc_port=WEAVIATE_GRPC_PORT)
    try:
        for col in (COL_PESSOA, COL_REGISTRO):
            client.collections.delete(col)
        print("Coleções removidas.")
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="POC de ponto com reconhecimento facial")
    parser.add_argument("--camera", type=int, default=0, help="índice da câmera (padrão 0)")
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("cadastrar", help="cadastra uma pessoa pela câmera")
    p.add_argument("--nome", required=True)
    p.add_argument("--amostras", type=int, default=5)
    p.set_defaults(func=cmd_cadastrar)

    p = sub.add_parser("monitorar", help="reconhece rostos e registra o ponto")
    p.add_argument("--limiar", type=float, default=0.30, help="distância cosine máxima")
    p.add_argument("--cooldown", type=int, default=60, help="segundos entre batidas da mesma pessoa")
    p.add_argument("--intervalo", type=float, default=0.5, help="segundos entre análises")
    p.set_defaults(func=cmd_monitorar)

    sub.add_parser("listar", help="lista pessoas cadastradas").set_defaults(func=cmd_listar)

    p = sub.add_parser("registros", help="lista batidas de ponto")
    p.add_argument("--dia", help="YYYY-MM-DD")
    p.add_argument("--limite", type=int, default=100)
    p.set_defaults(func=cmd_registros)

    p = sub.add_parser("remover", help="remove uma pessoa")
    p.add_argument("--nome", required=True)
    p.set_defaults(func=cmd_remover)

    sub.add_parser("reset", help="apaga todas as coleções").set_defaults(func=cmd_reset)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
