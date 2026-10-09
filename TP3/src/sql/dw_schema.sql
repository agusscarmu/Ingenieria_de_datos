-- Almacén de datos (OLAP) — esquema estrella del TP2, extendido para el TP3.
-- Grano: una fila por noticia-evento (GlobalEventID).
--
-- Extensiones respecto del TP2:
--   * DIM_ACTOR se usa con rol doble (actor1_key = origen, actor2_key = destino) e incorpora
--     el país del actor y su tipo (Type1Code) con su descripción del catálogo CAMEO.
--   * DIM_TIPO_INTERACCION (nueva): QuadClass desglosada en sus dos ejes
--     (cooperación/conflicto × verbal/material).
--   * DIM_UBICACION: región del evento (país FIPS + nombre, ADM1, tipo de geocodificación).
--   * FACT_COBERTURA_EVENTO: tono (avg_tone) e impacto estimado por Goldstein como medidas.
--   * Vista v_eventos_analisis: la tabla plana que alimenta el análisis multivariante.
-- DIM_GRUPO_CATEGORIA / BRIDGE_GRUPO_CAMEO del TP2 no se pueblan: el export.CSV trae un
-- único código CAMEO por evento.

PRAGMA foreign_keys = ON;

CREATE TABLE dim_tiempo (
    tiempo_key        INTEGER PRIMARY KEY,         -- AAAAMMDD
    fecha             TEXT    NOT NULL UNIQUE,
    anio              INTEGER NOT NULL,
    trimestre         INTEGER NOT NULL,
    mes               INTEGER NOT NULL,
    dia_mes           INTEGER NOT NULL,
    dia_semana        INTEGER NOT NULL,            -- 1 = lunes … 7
    es_fin_de_semana  INTEGER NOT NULL
);

CREATE TABLE dim_actor (
    actor_key         INTEGER PRIMARY KEY,         -- 0 = Sin actor
    actor_code        TEXT,
    nombre            TEXT,
    pais_actor_code   TEXT,                        -- CAMEO, 3 letras
    tipo_actor_code   TEXT,                        -- Type1Code
    tipo_actor_desc   TEXT,
    UNIQUE (actor_code, nombre, pais_actor_code, tipo_actor_code)
);

CREATE TABLE dim_cameo (
    cameo_key         INTEGER PRIMARY KEY,
    cameo_code        TEXT NOT NULL UNIQUE,
    descripcion       TEXT,
    codigo_base       TEXT,
    descripcion_base  TEXT,
    codigo_raiz       TEXT,
    descripcion_raiz  TEXT
);

CREATE TABLE dim_tipo_interaccion (
    interaccion_key   INTEGER PRIMARY KEY,         -- QuadClass 1..4
    descripcion       TEXT NOT NULL,
    eje_coop_conflicto TEXT NOT NULL,              -- Cooperación / Conflicto
    eje_verbal_material TEXT NOT NULL              -- Verbal / Material
);

CREATE TABLE dim_ubicacion (
    ubicacion_key     INTEGER PRIMARY KEY,         -- 0 = Sin ubicación
    pais_code         TEXT,                        -- FIPS 10-4
    pais_nombre       TEXT,
    adm1_code         TEXT,
    nombre_completo   TEXT,
    tipo_geo          INTEGER,
    latitud           REAL,
    longitud          REAL,
    UNIQUE (pais_code, adm1_code, nombre_completo, tipo_geo)
);

CREATE TABLE dim_medio (
    medio_key         INTEGER PRIMARY KEY,
    dominio           TEXT NOT NULL UNIQUE
);

CREATE TABLE fact_cobertura_evento (
    global_event_id        INTEGER PRIMARY KEY,    -- dimensión degenerada
    fecha_publicacion_key  INTEGER NOT NULL REFERENCES dim_tiempo (tiempo_key),
    fecha_evento_key       INTEGER NOT NULL REFERENCES dim_tiempo (tiempo_key),
    actor1_key             INTEGER NOT NULL REFERENCES dim_actor (actor_key),
    actor2_key             INTEGER NOT NULL REFERENCES dim_actor (actor_key),
    ubicacion_key          INTEGER NOT NULL REFERENCES dim_ubicacion (ubicacion_key),
    cameo_key              INTEGER NOT NULL REFERENCES dim_cameo (cameo_key),
    interaccion_key        INTEGER NOT NULL REFERENCES dim_tipo_interaccion (interaccion_key),
    medio_key              INTEGER NOT NULL REFERENCES dim_medio (medio_key),
    url_noticia            TEXT,
    es_evento_raiz         INTEGER NOT NULL,
    num_mentions           INTEGER NOT NULL,
    num_sources            INTEGER NOT NULL,
    num_articles           INTEGER NOT NULL,
    avg_tone               REAL    NOT NULL,       -- no aditiva
    goldstein_scale        REAL    NOT NULL        -- no aditiva (-10..10)
);

CREATE INDEX ix_fact_ubicacion   ON fact_cobertura_evento (ubicacion_key);
CREATE INDEX ix_fact_interaccion ON fact_cobertura_evento (interaccion_key);
CREATE INDEX ix_fact_cameo       ON fact_cobertura_evento (cameo_key);

CREATE VIEW v_eventos_analisis AS
SELECT
    f.global_event_id,
    tp.fecha                         AS fecha_publicacion,
    a1.actor_code                    AS actor1_code,
    CASE WHEN f.actor1_key = 0 THEN 'SIN_ACTOR'
         ELSE COALESCE(a1.tipo_actor_code, 'SIN_TIPO') END AS actor1_tipo,
    a2.actor_code                    AS actor2_code,
    CASE WHEN f.actor2_key = 0 THEN 'SIN_ACTOR'
         ELSE COALESCE(a2.tipo_actor_code, 'SIN_TIPO') END AS actor2_tipo,
    ti.descripcion                   AS tipo_interaccion,
    c.cameo_code,
    c.codigo_raiz || ' ' || c.descripcion_raiz AS evento_raiz,
    COALESCE(u.pais_code, 'SIN_UBICACION')     AS pais_accion,
    u.pais_nombre,
    u.adm1_code,
    u.latitud,
    u.longitud,
    f.es_evento_raiz,
    f.goldstein_scale,
    f.avg_tone,
    f.num_mentions,
    f.num_sources,
    f.num_articles
FROM fact_cobertura_evento f
JOIN dim_tiempo           tp ON tp.tiempo_key      = f.fecha_publicacion_key
JOIN dim_actor            a1 ON a1.actor_key       = f.actor1_key
JOIN dim_actor            a2 ON a2.actor_key       = f.actor2_key
JOIN dim_tipo_interaccion ti ON ti.interaccion_key = f.interaccion_key
JOIN dim_cameo            c  ON c.cameo_key        = f.cameo_key
JOIN dim_ubicacion        u  ON u.ubicacion_key    = f.ubicacion_key;
