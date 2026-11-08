# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Quando e como comprar o voo mais barato?
# MAGIC ### Projeto de Big Data com Apache Spark e Databricks Free Edition
# MAGIC
# MAGIC **Grupo:** _nome (nº), nome (nº), nome (nº), nome (nº), nome (nº)_
# MAGIC
# MAGIC **Índice**
# MAGIC 1. Problema e motivação
# MAGIC 2. Dados
# MAGIC 3. Arquitetura
# MAGIC 4. Configuração
# MAGIC 5. Bronze: ingestão em streaming
# MAGIC 6. Silver: limpeza e enriquecimento
# MAGIC 7. Gold e análise: as três perguntas
# MAGIC 8. Funcionalidades Delta Lake
# MAGIC 9. Conclusões

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Problema e motivação
# MAGIC
# MAGIC O preço de um mesmo voo muda todos os dias. Metabuscadores como a **Skyscanner** existem para tornar essa
# MAGIC variação visível ao viajante, e fazem-no com Big Data: a Skyscanner processa dezenas de milhões de pesquisas por
# MAGIC dia e usa a plataforma Databricks.
# MAGIC
# MAGIC Os dados da Skyscanner não são públicos (a API é só para parceiros comerciais e não permite guardar preços).
# MAGIC Usamos por isso um dataset público equivalente: **milhões de preços de voos pesquisados diariamente na Expedia**,
# MAGIC exatamente o tipo de dado que um metabuscador vê.
# MAGIC
# MAGIC **Perguntas de investigação**
# MAGIC
# MAGIC | # | Pergunta | Porque interessa ao viajante |
# MAGIC |---|---|---|
# MAGIC | P1 | **Com quantos dias de antecedência compensa comprar?** | É a decisão mais comum: comprar já ou esperar |
# MAGIC | P2 | **Quanto custa a mais um voo direto face a um com escala?** | Mede o preço do conforto e do tempo poupado |
# MAGIC | P3 | **Que companhias são mais baratas por milha voada?** | Compara companhias em rotas de distâncias diferentes |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Dados
# MAGIC
# MAGIC | Item | Valor |
# MAGIC |---|---|
# MAGIC | Dataset | Flight Prices, Kaggle (`dilwong/flightprices`) |
# MAGIC | Origem | Pesquisas na Expedia de voos de ida, 16 abr a 5 out 2022, 16 aeroportos dos EUA |
# MAGIC | Volume | ≈31 GB em CSV, dezenas de milhões de linhas |
# MAGIC | Granularidade | Uma linha por voo (`legId`) por dia de pesquisa (`searchDate`) |
# MAGIC | Tipo | Estruturado, com campos multi-valor (um valor por segmento, separados por `\|\|`) |
# MAGIC | Streaming | Cada dia de pesquisa chega como um ficheiro novo, lido com Auto Loader |
# MAGIC
# MAGIC **Preparação fora do Databricks (feita uma vez):** o CSV foi convertido para Parquet, partido por dia de pesquisa,
# MAGIC com DuckDB, e carregado para o Volume `workspace.raw.landing/flights_all/`:
# MAGIC
# MAGIC ```sql
# MAGIC COPY (SELECT * FROM read_csv_auto('itineraries.csv'))
# MAGIC TO 'flights_all' (FORMAT PARQUET, PARTITION_BY (searchDate), COMPRESSION ZSTD);
# MAGIC ```

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.raw;
# MAGIC CREATE VOLUME IF NOT EXISTS workspace.raw.landing;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Arquitetura
# MAGIC
# MAGIC Arquitetura **Medallion** em Delta Lake, organizada no Unity Catalog:
# MAGIC
# MAGIC ```
# MAGIC  Volume landing/flights_stream/            (um dia de pesquisa = uma pasta nova)
# MAGIC          │  Auto Loader, trigger availableNow
# MAGIC          ▼
# MAGIC  BRONZE  bronze.flights        dados em bruto + metadados de ingestão, partido por dia
# MAGIC          │  tipos, nomes, colunas derivadas, regras de qualidade
# MAGIC          ▼
# MAGIC  SILVER  silver.voos           1 linha por voo e dia de pesquisa
# MAGIC          silver.segmentos      1 linha por segmento de voo (explode dos campos ||)
# MAGIC          │  agregações
# MAGIC          ▼
# MAGIC  GOLD    gold.p1_antecedencia, gold.p1_mesmo_voo, gold.p2_direto_vs_escala, gold.p3_companhias
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Configuração
# MAGIC
# MAGIC Constantes e funções usadas em todo o notebook. `DEV_DIAS` limita o número de dias processados durante o
# MAGIC desenvolvimento (para respeitar as quotas do Free Edition); no Run all final fica a `None` para usar tudo.

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "workspace"
LANDING = "/Volumes/workspace/raw/landing"
ORIGEM_DIAS = f"{LANDING}/flights_all"       # todos os dias convertidos
PASTA_STREAM = f"{LANDING}/flights_stream"   # pasta vigiada pelo Auto Loader
CHECKPOINTS = f"{LANDING}/checkpoints"

DEV_DIAS = 7            # 7 durante o desenvolvimento; None no Run all final
MAX_ANTECEDENCIA = 60   # dias de antecedência analisados na P1
MIN_OBS = 1_000         # mínimo de observações para um grupo entrar nos resultados

spark.sql(f"USE CATALOG {CATALOG}")
for schema in ["raw", "bronze", "silver", "gold"]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
spark.sql("CREATE VOLUME IF NOT EXISTS raw.landing")


def guardar(df: DataFrame, tabela: str, particao: str = None) -> None:
    """Grava um DataFrame como tabela Delta (substitui a anterior) e mostra o número de linhas."""
    escritor = df.write.mode("overwrite").option("overwriteSchema", True)
    if particao:
        escritor = escritor.partitionBy(particao)
    escritor.saveAsTable(tabela)
    print(f"{tabela}: {spark.table(tabela).count():,} linhas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Bronze: ingestão em streaming
# MAGIC
# MAGIC Simulamos o que um metabuscador recebe: **um dia de pesquisas de cada vez**. A função `publicar_dias` copia os
# MAGIC próximos dias de `flights_all/` para a pasta vigiada. O **Auto Loader** (`cloudFiles`) lê só os ficheiros que
# MAGIC ainda não processou, graças ao checkpoint. No serverless só é suportado o trigger `availableNow`: cada execução
# MAGIC processa o que está pendente e termina.
# MAGIC
# MAGIC Na Bronze nada é limpo: guardamos os dados como vieram, mais `_source_file` e `_ingest_ts` para rastreabilidade.

# COMMAND ----------

dbutils.fs.mkdirs(PASTA_STREAM)
TODOS_DIAS = sorted(f.name.rstrip("/") for f in dbutils.fs.ls(ORIGEM_DIAS) if f.name.startswith("searchDate="))
DIAS_A_USAR = TODOS_DIAS[:DEV_DIAS] if DEV_DIAS else TODOS_DIAS
print(f"{len(TODOS_DIAS)} dias disponíveis, {len(DIAS_A_USAR)} a usar")


def publicar_dias(n: int) -> list:
    """Copia os próximos n dias de pesquisa para a pasta vigiada pelo stream (simula a chegada de dados)."""
    ja_publicados = {f.name.rstrip("/") for f in dbutils.fs.ls(PASTA_STREAM)}
    novos = [d for d in DIAS_A_USAR if d not in ja_publicados][:n]
    for dia in novos:
        dbutils.fs.cp(f"{ORIGEM_DIAS}/{dia}", f"{PASTA_STREAM}/{dia}", recurse=True)
    return novos


def ingerir_bronze() -> None:
    """Lê com Auto Loader os ficheiros novos e acrescenta-os a bronze.flights."""
    (spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.schemaLocation", f"{CHECKPOINTS}/flights_schema")
        .option("cloudFiles.partitionColumns", "searchDate")
        .load(PASTA_STREAM)
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_ingest_ts", F.current_timestamp())
     .writeStream
        .option("checkpointLocation", f"{CHECKPOINTS}/flights_bronze")
        .partitionBy("searchDate")
        .trigger(availableNow=True)
        .toTable("bronze.flights")
        .awaitTermination())

# COMMAND ----------

# MAGIC %md
# MAGIC **Demonstração do streaming.** Publicamos primeiro 3 dias e corremos o stream; depois publicamos os restantes e
# MAGIC corremos outra vez. Na segunda execução só os dias novos são lidos.

# COMMAND ----------

print("1.º lote:", publicar_dias(3))
ingerir_bronze()
display(spark.sql("SELECT searchDate, COUNT(*) AS linhas FROM bronze.flights GROUP BY searchDate ORDER BY searchDate"))

# COMMAND ----------

print("2.º lote:", len(publicar_dias(len(DIAS_A_USAR))), "dias")
ingerir_bronze()
display(spark.sql("""
    SELECT COUNT(DISTINCT searchDate) AS dias, COUNT(*) AS linhas, COUNT(DISTINCT _source_file) AS ficheiros
    FROM bronze.flights
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Silver: limpeza e enriquecimento
# MAGIC
# MAGIC Transformações aplicadas a `bronze.flights`:
# MAGIC
# MAGIC | Passo | O quê | Porquê |
# MAGIC |---|---|---|
# MAGIC | Tipos | datas como `date`, preços como `double`, flags como `boolean` | Os dados chegam como texto |
# MAGIC | Nomes | snake_case em português | Legibilidade e consistência |
# MAGIC | `dias_antecedencia` | `data_voo − data_pesquisa` | Base da P1 |
# MAGIC | `duracao_min` | `PT5H30M` → 330 | Formato ISO 8601 não é numérico |
# MAGIC | `escalas` | número de segmentos − 1 | Base da P2 |
# MAGIC | Campos `\|\|` | texto → arrays | Um valor por segmento do voo |
# MAGIC | `distancia_segmentos_total` | soma das distâncias dos segmentos | Base da P3 |
# MAGIC | Qualidade | preço > 0, antecedência ≥ 0, linhas repetidas removidas; `try_cast` em vez de `cast` | Remover registos impossíveis sem o modo ANSI dar erro |

# COMMAND ----------

def para_minutos(col_duracao: str):
    """Converte uma duração ISO 8601 (ex.: 'PT5H30M') em minutos."""
    horas = F.regexp_extract(F.col(col_duracao), r"(\d+)H", 1)
    minutos = F.regexp_extract(F.col(col_duracao), r"(\d+)M", 1)
    return (F.coalesce(F.when(horas != "", horas.cast("int")), F.lit(0)) * 60
            + F.coalesce(F.when(minutos != "", minutos.cast("int")), F.lit(0)))


def separar(col_segmentos: str):
    """Transforma um campo 'a||b||c' num array [a, b, c]."""
    return F.split(F.col(col_segmentos), r"\|\|")


def numero(col: str):
    """Converte para double; valores inválidos (ex.: 'None') ficam nulos em vez de dar erro (modo ANSI)."""
    return F.expr(f"try_cast({col} AS DOUBLE)")


def construir_silver_voos(bronze: DataFrame) -> DataFrame:
    """Limpa e enriquece os voos: uma linha por voo e por dia de pesquisa."""
    return (bronze
        .select(
            "legId",
            F.to_date("searchDate").alias("data_pesquisa"),
            F.to_date("flightDate").alias("data_voo"),
            F.col("startingAirport").alias("origem"),
            F.col("destinationAirport").alias("destino"),
            numero("totalFare").alias("preco_usd"),
            F.col("seatsRemaining").cast("int").alias("lugares_restantes"),
            F.col("isNonStop").cast("boolean").alias("direto"),
            F.col("isBasicEconomy").cast("boolean").alias("basic_economy"),
            numero("totalTravelDistance").alias("distancia_milhas"),
            para_minutos("travelDuration").alias("duracao_min"),
            separar("segmentsAirlineName").alias("companhias"),
            F.expr(r"transform(split(segmentsDistance, '\\|\\|'), x -> try_cast(x AS DOUBLE))")
             .alias("distancias_segmento"))
        .withColumn("distancia_segmentos_total",
                    F.expr("aggregate(distancias_segmento, 0D, (soma, x) -> soma + coalesce(x, 0D))"))
        .withColumn("rota", F.concat_ws("-", "origem", "destino"))
        .withColumn("dias_antecedencia", F.datediff("data_voo", "data_pesquisa"))
        .withColumn("escalas", F.size("companhias") - 1)
        .filter((F.col("preco_usd") > 0) & (F.col("dias_antecedencia") >= 0))
        .dropDuplicates())

# COMMAND ----------

guardar(construir_silver_voos(spark.table("bronze.flights")), "silver.voos", particao="data_pesquisa")
display(spark.table("silver.voos").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC **Segmentos.** Um voo com escala tem vários segmentos, cada um com a sua companhia e distância. Para comparar
# MAGIC companhias (P3) usamos `arrays_zip` + `explode`: cada voo dá origem a uma linha por segmento. O preço é
# MAGIC repartido pelos segmentos proporcionalmente à distância de cada um.

# COMMAND ----------

def construir_silver_segmentos(voos: DataFrame) -> DataFrame:
    """Uma linha por segmento: companhia, distância e quota do preço proporcional à distância."""
    return (voos
        .filter(F.col("distancia_segmentos_total") > 0)
        .select("legId", "data_pesquisa", "rota", "preco_usd", "escalas", "distancia_segmentos_total",
                F.explode(F.arrays_zip("companhias", "distancias_segmento")).alias("seg"))
        .select("legId", "data_pesquisa", "rota", "preco_usd", "escalas",
                F.trim(F.col("seg.companhias")).alias("companhia"),
                F.col("seg.distancias_segmento").alias("distancia_milhas"),
                (F.col("preco_usd") * F.col("seg.distancias_segmento")
                 / F.col("distancia_segmentos_total")).alias("preco_segmento_usd"))
        .filter(F.col("distancia_milhas") > 0))


guardar(construir_silver_segmentos(spark.table("silver.voos")), "silver.segmentos")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Gold e análise
# MAGIC
# MAGIC **Nota:** o mesmo voo pode aparecer várias vezes no mesmo dia, com tarifas diferentes (`fareBasisCode`).
# MAGIC Nas análises por voo usamos a tarifa mais barata.
# MAGIC
# MAGIC **Normalização por rota.** Um voo Nova Iorque–Los Angeles é sempre mais caro que um Boston–Nova Iorque. Para
# MAGIC comparar rotas diferentes dividimos cada preço pela **mediana da sua rota**: um índice de 1,20 significa
# MAGIC "20% acima do normal nessa rota". Usamos medianas em vez de médias porque os preços têm valores extremos.

# COMMAND ----------

mediana_rota = (spark.table("silver.voos")
                .groupBy("rota")
                .agg(F.percentile_approx("preco_usd", 0.5).alias("mediana_rota_usd")))

guardar(spark.table("silver.voos").join(mediana_rota, "rota")
        .withColumn("indice_preco", F.col("preco_usd") / F.col("mediana_rota_usd")),
        "gold.voos_indice")

# COMMAND ----------

# MAGIC %md
# MAGIC ### P1. Com quantos dias de antecedência compensa comprar?
# MAGIC
# MAGIC **Abordagem 1, visão geral:** índice de preço mediano para cada número de dias de antecedência (0 a 60).

# COMMAND ----------

p1 = (spark.table("gold.voos_indice")
      .filter(F.col("dias_antecedencia") <= MAX_ANTECEDENCIA)
      .groupBy("dias_antecedencia")
      .agg(F.percentile_approx("indice_preco", 0.5).alias("indice_mediano"),
           F.count("*").alias("n_observacoes"))
      .orderBy("dias_antecedencia"))
guardar(p1, "gold.p1_antecedencia")

display(spark.table("gold.p1_antecedencia"))   # Visualização: linha, x = dias_antecedencia, y = indice_mediano

# COMMAND ----------

# MAGIC %md
# MAGIC **Abordagem 2, o mesmo voo ao longo do tempo:** seguimos cada voo (`legId`) de pesquisa em pesquisa com a
# MAGIC window function `lag`, como faz um alerta de preço. Para cada antecedência calculamos a **probabilidade de o
# MAGIC preço subir** face à pesquisa anterior e a variação média em dólares. Isto responde diretamente a "se eu esperar
# MAGIC mais um dia, o preço sobe?".

# COMMAND ----------

por_voo = Window.partitionBy("legId").orderBy("data_pesquisa")

p1_mesmo_voo = (spark.table("silver.voos")
    .groupBy("legId", "data_pesquisa", "dias_antecedencia")          # um preço por voo e dia:
    .agg(F.min("preco_usd").alias("preco_usd"))                       # a tarifa mais barata
    .withColumn("preco_anterior", F.lag("preco_usd").over(por_voo))
    .withColumn("pesquisa_anterior", F.lag("data_pesquisa").over(por_voo))
    .filter(F.datediff("data_pesquisa", "pesquisa_anterior") == 1)   # só pesquisas em dias seguidos
    .withColumn("variacao_usd", F.col("preco_usd") - F.col("preco_anterior"))
    .filter(F.col("dias_antecedencia") <= MAX_ANTECEDENCIA)
    .groupBy("dias_antecedencia")
    .agg(F.round(F.avg((F.col("variacao_usd") > 0).cast("int")) * 100, 1).alias("pct_subiu"),
         F.round(F.avg((F.col("variacao_usd") < 0).cast("int")) * 100, 1).alias("pct_desceu"),
         F.round(F.avg("variacao_usd"), 2).alias("variacao_media_usd"),
         F.count("*").alias("n_observacoes"))
    .orderBy("dias_antecedencia"))
guardar(p1_mesmo_voo, "gold.p1_mesmo_voo")

display(spark.table("gold.p1_mesmo_voo"))   # Visualização: linhas pct_subiu e pct_desceu por dias_antecedencia

# COMMAND ----------

# MAGIC %md
# MAGIC **Resumo da P1 em SQL:** comparamos três janelas de compra.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT CASE WHEN dias_antecedencia <= 7  THEN '1. Última semana (0-7 dias)'
# MAGIC             WHEN dias_antecedencia <= 21 THEN '2. 1 a 3 semanas (8-21 dias)'
# MAGIC             ELSE                              '3. Mais de 3 semanas (22-60 dias)' END AS janela_compra,
# MAGIC        ROUND(PERCENTILE_APPROX(indice_preco, 0.5), 3) AS indice_mediano,
# MAGIC        COUNT(*) AS n_observacoes
# MAGIC FROM gold.voos_indice
# MAGIC WHERE dias_antecedencia <= 60
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1

# COMMAND ----------

# MAGIC %md
# MAGIC _Leitura (preencher depois de correr): "O índice desce de X na última semana para Y a partir de Z dias de
# MAGIC antecedência. Na última semana o preço do mesmo voo sobe de um dia para o outro em W% dos casos."_

# COMMAND ----------

# MAGIC %md
# MAGIC ### P2. Quanto custa a mais um voo direto face a um com escala?
# MAGIC
# MAGIC Comparamos **na mesma rota e no mesmo dia de pesquisa**: o preço mediano dos voos diretos dividido pelo dos voos
# MAGIC com escala. Assim a diferença não vem de rotas ou datas diferentes. Só entram rotas que têm os dois tipos de voo.
# MAGIC Mostramos também quantos minutos o voo direto poupa.

# COMMAND ----------

por_rota_dia = (spark.table("silver.voos")
    .groupBy("rota", "data_pesquisa", "direto")
    .agg(F.percentile_approx("preco_usd", 0.5).alias("preco_mediano"),
         F.percentile_approx("duracao_min", 0.5).alias("duracao_mediana"))
    .groupBy("rota", "data_pesquisa")
    .pivot("direto", [True, False])
    .agg(F.first("preco_mediano").alias("preco"), F.first("duracao_mediana").alias("duracao"))
    .withColumnRenamed("true_preco", "preco_direto").withColumnRenamed("false_preco", "preco_escala")
    .withColumnRenamed("true_duracao", "duracao_direto").withColumnRenamed("false_duracao", "duracao_escala")
    .dropna(subset=["preco_direto", "preco_escala"]))

p2 = (por_rota_dia
    .groupBy("rota")
    .agg(F.round(F.percentile_approx(F.col("preco_direto") / F.col("preco_escala"), 0.5), 3).alias("racio_direto_escala"),
         F.round(F.percentile_approx(F.col("preco_direto") - F.col("preco_escala"), 0.5), 2).alias("diferenca_usd"),
         F.round(F.percentile_approx(F.col("duracao_escala") - F.col("duracao_direto"), 0.5), 0).alias("minutos_poupados"),
         F.count("*").alias("n_dias"))
    .withColumn("usd_por_hora_poupada",
                F.when(F.col("minutos_poupados") > 0,
                       F.round(F.col("diferenca_usd") / (F.col("minutos_poupados") / 60), 2)))
    .orderBy(F.desc("racio_direto_escala")))
guardar(p2, "gold.p2_direto_vs_escala")

display(spark.table("gold.p2_direto_vs_escala"))   # Visualização: barras horizontais, rota × racio_direto_escala

# COMMAND ----------

display(spark.table("gold.p2_direto_vs_escala").agg(
    F.round(F.percentile_approx("racio_direto_escala", 0.5), 3).alias("racio_mediano_todas_rotas"),
    F.round(F.percentile_approx("diferenca_usd", 0.5), 2).alias("diferenca_mediana_usd"),
    F.round(F.percentile_approx("minutos_poupados", 0.5), 0).alias("minutos_poupados_medianos")))

# COMMAND ----------

# MAGIC %md
# MAGIC _Leitura: "Na rota mediana, o voo direto custa X% mais (Y USD) e poupa Z minutos, ou seja W USD por hora
# MAGIC poupada. As rotas onde o direto é mais caro são…; em algumas rotas o direto é até mais barato."_

# COMMAND ----------

# MAGIC %md
# MAGIC ### P3. Que companhias são mais baratas por milha voada?
# MAGIC
# MAGIC Usamos `silver.segmentos` (uma linha por segmento), para que voos com escala em companhias diferentes contem
# MAGIC para cada uma. O indicador é o **preço por milha** de cada segmento. Classificamos as companhias com a window
# MAGIC function `rank`, e só entram companhias com pelo menos `MIN_OBS` segmentos.
# MAGIC
# MAGIC Como o preço por milha é naturalmente mais alto em voos curtos, mostramos também o resultado por faixa de
# MAGIC distância, para a comparação ser justa.

# COMMAND ----------

segmentos = (spark.table("silver.segmentos")
    .withColumn("usd_por_milha", F.col("preco_segmento_usd") / F.col("distancia_milhas"))
    .withColumn("faixa_distancia",
                F.when(F.col("distancia_milhas") < 500, "1. Curta (<500 mi)")
                 .when(F.col("distancia_milhas") < 1500, "2. Média (500-1500 mi)")
                 .otherwise("3. Longa (>1500 mi)")))

p3 = (segmentos
    .groupBy("companhia")
    .agg(F.round(F.percentile_approx("usd_por_milha", 0.5), 4).alias("usd_por_milha"),
         F.round(F.percentile_approx("distancia_milhas", 0.5), 0).alias("distancia_mediana_milhas"),
         F.count("*").alias("n_segmentos"))
    .filter(F.col("n_segmentos") >= MIN_OBS)
    .withColumn("ranking", F.rank().over(Window.orderBy("usd_por_milha")))
    .orderBy("ranking"))
guardar(p3, "gold.p3_companhias")

display(spark.table("gold.p3_companhias"))   # Visualização: barras horizontais, companhia × usd_por_milha

# COMMAND ----------

p3_faixa = (segmentos
    .groupBy("faixa_distancia", "companhia")
    .agg(F.round(F.percentile_approx("usd_por_milha", 0.5), 4).alias("usd_por_milha"),
         F.count("*").alias("n_segmentos"))
    .filter(F.col("n_segmentos") >= MIN_OBS)
    .withColumn("ranking_na_faixa",
                F.rank().over(Window.partitionBy("faixa_distancia").orderBy("usd_por_milha")))
    .orderBy("faixa_distancia", "ranking_na_faixa"))
guardar(p3_faixa, "gold.p3_companhias_por_faixa")

display(spark.table("gold.p3_companhias_por_faixa"))   # Visualização: barras agrupadas por faixa_distancia

# COMMAND ----------

# MAGIC %md
# MAGIC _Leitura: "A companhia mais barata por milha é X (Y USD/milha), a mais cara é Z. Nas três faixas de distância a
# MAGIC ordem mantém-se / muda em…"_

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Funcionalidades Delta Lake
# MAGIC
# MAGIC Cada execução do stream criou uma versão nova de `bronze.flights`. O histórico permite **time travel**:
# MAGIC consultar a tabela tal como estava depois do primeiro lote. O `OPTIMIZE ... ZORDER BY` reorganiza os ficheiros
# MAGIC para acelerar filtros por rota.

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY bronze.flights

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'versão 0 (1.º lote)' AS versao, COUNT(*) AS linhas FROM bronze.flights VERSION AS OF 0
# MAGIC UNION ALL
# MAGIC SELECT 'versão atual', COUNT(*) FROM bronze.flights

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE silver.voos ZORDER BY (rota)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Conclusões
# MAGIC
# MAGIC _Escrever um parágrafo por pergunta com os números obtidos, como conselho ao viajante._
# MAGIC
# MAGIC - **P1:** …
# MAGIC - **P2:** …
# MAGIC - **P3:** …
# MAGIC
# MAGIC **Limitações**
# MAGIC - Dados da Expedia (não da Skyscanner), voos domésticos nos EUA, abril a outubro de 2022.
# MAGIC - Preços de ida, para um passageiro; não incluem bagagem nem outras taxas posteriores.
# MAGIC - Na P3 o preço de um voo com escala é repartido pelos segmentos de forma proporcional à distância, o que é
# MAGIC   uma aproximação.
# MAGIC - Correlação não é causalidade: a antecedência e o preço dependem ambos da procura.
# MAGIC
# MAGIC **Tecnologias usadas:** Apache Spark (DataFrames, Spark SQL, window functions), Delta Lake (tabelas, time
# MAGIC travel, OPTIMIZE), Auto Loader e Structured Streaming, Unity Catalog (schemas e Volumes).