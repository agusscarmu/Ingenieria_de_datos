-- Almacén del TP: esquema estrella con las ventanas de 30 s de los Holter de SHDB-AF.
-- Lo ejecuta almacen.construir(), que antes activa PRAGMA foreign_keys = ON: SQLite trae las claves foráneas
-- apagadas y se prenden por conexión, no desde el script.
--
-- Estrella: en el centro, hecho_ventana (una fila por ventana de 30 s que se pudo analizar) y alrededor sus tres
-- dimensiones: dim_holter (de quién es la ventana), dim_tiempo (a qué hora del día) y dim_ritmo (en qué ritmo estaba).
-- Las dimensiones están desnormalizadas a propósito: cada una junta en una sola tabla lo que en la base transaccional
-- está repartido, para que el tablero arme sus consultas con un JOIN por dimensión.

-- Dimensiones

-- Una fila por Holter. Reúne paciente, holter y tipo_fa de la transaccional; talla y peso no se llevan.
CREATE TABLE dim_holter (
    holter_key  INTEGER PRIMARY KEY,                        -- clave sustituta: 1..n, en el orden de holter_id
    holter_id   INTEGER NOT NULL UNIQUE,                    -- Data_ID del CSV
    paciente_id INTEGER NOT NULL,
    sexo        TEXT    NOT NULL CHECK (sexo IN ('F', 'M')),
    edad        INTEGER NOT NULL CHECK (edad >= 0),         -- años al hacer el Holter
    tipo_fa     TEXT    NOT NULL CHECK (tipo_fa IN ('PAF', 'PerAF', 'non-AF')),
    anotado     INTEGER NOT NULL CHECK (anotado IN (0, 1))  -- 1 si tiene .atr (episodios de ritmo)
);

-- Una fila por cada ventana posible del día: 24 h / 30 s = 2.880. Es hora del día sin fecha (las fechas de la base
-- están corridas): en las ventanas posteriores a la medianoche, también en las del segundo día de un Holter
-- largo, la hora vuelve a empezar en 00:00:00.
CREATE TABLE dim_tiempo (
    tiempo_key  INTEGER PRIMARY KEY CHECK (tiempo_key BETWEEN 0 AND 2879),  -- segundos desde la medianoche / 30
    hora_inicio TEXT    NOT NULL UNIQUE CHECK (length(hora_inicio) = 8 AND time(hora_inicio) IS NOT NULL),  -- hh:mm:ss
    hora        INTEGER NOT NULL CHECK (hora BETWEEN 0 AND 23),         -- hora entera de hora_inicio
    franja      TEXT    NOT NULL CHECK (franja IN ('día', 'noche'))     -- noche: hora >= 23 o < 7; día: el resto
);

-- Los dos ritmos con los que se comparan las ventanas. Las que no son ni una ni otra no llegan a los hechos.
CREATE TABLE dim_ritmo (
    ritmo_key INTEGER PRIMARY KEY,
    codigo    TEXT NOT NULL UNIQUE CHECK (codigo IN ('AFIB', 'N')),     -- como en el .atr
    etiqueta  TEXT NOT NULL UNIQUE                                      -- el nombre que se muestra
);

-- Hechos

-- Grano: una fila por ventana de 30 s con un solo ritmo (todos sus latidos AFIB, o todos N) y al menos 5 RR. Las
-- ventanas descartadas por mezcla de ritmos o por pocos RR no están acá.
-- (holter_key, n_ventana) es la clave: n_ventana, el número de la ventana en su Holter (desde 0), es una dimensión
-- degenerada, sin tabla propia. Las 17 características son las de caracteristicas.py y quedan nulas, sin imputar,
-- cuando la ventana no tiene con qué calcularlas.
-- WITHOUT ROWID: la fila vive en el índice de la clave, así que las ventanas de un Holter quedan juntas y en orden.
CREATE TABLE hecho_ventana (
    holter_key INTEGER NOT NULL REFERENCES dim_holter (holter_key),
    n_ventana  INTEGER NOT NULL CHECK (n_ventana >= 0),
    tiempo_key INTEGER NOT NULL REFERENCES dim_tiempo (tiempo_key),     -- hora del día en que empieza la ventana
    ritmo_key  INTEGER NOT NULL REFERENCES dim_ritmo (ritmo_key),
    n_rr       INTEGER NOT NULL CHECK (n_rr >= 5),                      -- RR de la ventana; 5 es el mínimo (MIN_RR)
    P_cv       REAL,
    P_RMSSD    REAL,
    P_NMASD    REAL,
    P_IPD      REAL,
    PNN50      REAL,
    PNN100     REAL,
    P_SamEn    REAL,
    P_SSamEn   REAL,
    P_CSamEn   REAL,
    P_ShEn     REAL,
    P_Shan2    REAL,
    P_NZPP1    REAL,
    P_NZPP2    REAL,
    P_NZPP3    REAL,
    P_sigma_y0 REAL,
    P_sigma_y1 REAL,
    P_sigma_ED REAL,
    PRIMARY KEY (holter_key, n_ventana)
) WITHOUT ROWID;
