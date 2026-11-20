-- Base transaccional del TP: Holter de SHDB-AF en 3FN.
-- La ejecuta transaccional.crear(), que antes activa PRAGMA foreign_keys = ON: SQLite trae las claves
-- foráneas apagadas y se prenden por conexión, no desde el script.
--
-- Relaciones: un paciente tiene varios Holter, y cada Holter tiene un tipo de FA y muchos latidos.
-- Un episodio de ritmo empieza en un latido y tiene un ritmo.

-- Catálogos: el código y su descripción (la descripción depende del código, no de cada Holter o episodio)

CREATE TABLE tipo_fa (
    tipo_fa     TEXT NOT NULL PRIMARY KEY,      -- PAF, PerAF, non-AF
    descripcion TEXT NOT NULL
);

CREATE TABLE ritmo (
    ritmo       TEXT NOT NULL PRIMARY KEY,      -- código del .atr sin "(": AFIB, N, AFL, AT, AB, PAT, NOD
    descripcion TEXT NOT NULL
);

-- Pacientes y Holter

CREATE TABLE paciente (
    paciente_id INTEGER PRIMARY KEY,                        -- Subject_ID del CSV
    sexo        TEXT NOT NULL CHECK (sexo IN ('F', 'M'))    -- del paciente, no del Holter: es la 3FN
);

-- El CHECK de tipo_fa repite el catálogo a propósito: en SQLite las claves foráneas están apagadas
-- salvo que la conexión las prenda, y el CHECK vale siempre.
CREATE TABLE holter (
    holter_id   INTEGER PRIMARY KEY,                        -- Data_ID del CSV
    paciente_id INTEGER NOT NULL REFERENCES paciente (paciente_id),
    tipo_fa     TEXT    NOT NULL REFERENCES tipo_fa (tipo_fa)
                        CHECK (tipo_fa IN ('PAF', 'PerAF', 'non-AF')),
    anotado     INTEGER NOT NULL CHECK (anotado IN (0, 1)), -- 1 si tiene .atr (episodios de ritmo)
    edad        INTEGER NOT NULL CHECK (edad >= 0),         -- años al hacer el Holter
    talla_m     REAL,                                       -- faltantes: quedan nulos, sin imputar
    peso_kg     REAL,
    hora_inicio TEXT    NOT NULL                            -- HH:MM, 24 h: de 00:00 a 23:59
                        CHECK (hora_inicio GLOB '[01][0-9]:[0-5][0-9]'
                               OR hora_inicio GLOB '2[0-3]:[0-5][0-9]'),
    fs_hz       INTEGER NOT NULL CHECK (fs_hz > 0),         -- frecuencia de muestreo, del .hea
    n_muestras  INTEGER NOT NULL CHECK (n_muestras > 0)     -- largo del registro, del .hea
);

-- Un latido por fila. El RR no se guarda porque se calcula de dos latidos seguidos.
-- WITHOUT ROWID: la fila vive en el índice de la clave (holter_id, n), sin tabla aparte. Los latidos
-- de un Holter quedan juntos y en orden, que es como se leen y se cargan.

CREATE TABLE latido (
    holter_id INTEGER NOT NULL REFERENCES holter (holter_id),
    n         INTEGER NOT NULL CHECK (n >= 0),          -- orden en el Holter, desde 0
    muestra   INTEGER NOT NULL CHECK (muestra >= 0),    -- posición del latido en la señal, en muestras
    PRIMARY KEY (holter_id, n)
) WITHOUT ROWID;

-- Un episodio por cada marca de ritmo del .atr: el ritmo vale desde ese latido hasta el próximo
-- episodio del Holter. Los Holter sin .atr no tienen filas acá.

CREATE TABLE episodio_ritmo (
    holter_id     INTEGER NOT NULL,
    latido_inicio INTEGER NOT NULL,                         -- n del latido donde empieza el ritmo
    ritmo         TEXT    NOT NULL REFERENCES ritmo (ritmo),
    PRIMARY KEY (holter_id, latido_inicio),
    FOREIGN KEY (holter_id, latido_inicio) REFERENCES latido (holter_id, n)
);
