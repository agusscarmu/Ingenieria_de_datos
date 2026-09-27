## ¿Qué guarda la base?

`holter.db` reúne la información de los estudios Holter del ICBA usados en la tesis sobre
fibrilación auricular (FA). Las señales ECG y los PDF de los informes **no** se guardan dentro
de la base: quedan en sus carpetas y la base registra dónde están.

### Datos principales
| Tabla | Qué guarda | Por qué |
|---|---|---|
| `paciente` | Una fila por persona: DNI, historia clínica, nombre, sexo, fecha de nacimiento. | Un mismo paciente puede tener varios Holter; permite no mezclar pacientes al separar datos de entrenamiento y prueba. |
| `estudio` | Un Holter (~56.000, toda la base del equipo): fecha, hora de inicio, latidos, extrasístoles, pausas, FC media/mín/máx. | Es la unidad de análisis. |
| `senal` | La grabación ECG de un estudio: dónde está, frecuencia de muestreo, duración y si está completa. | Saber qué estudios tienen señal utilizable. |
| `senal_canal` | Cada uno de los 3 canales de la grabación: archivo, ganancia, calidad. | Detectar canales faltantes o de mala calidad. |
| `informe` | El informe PDF del cardiólogo y su texto. | Es la fuente del diagnóstico. |
| `etiqueta` | El diagnóstico de FA de cada informe según cada fuente (conclusión del informe, sección de ESV y, más adelante, cardiólogos y modelos). | Poder comparar diagnósticos de distintas fuentes. |

### Catálogos
| Tabla | Qué guarda |
|---|---|
| `categoria_fa` | Categorías posibles: sin FA, FA paroxística, FA no paroxística, indeterminado. |
| `fuente_etiqueta` | Quién o qué asignó cada etiqueta. |
| `tipo_observacion` | Tipos de problema de datos que se registran. |

### Control de calidad y trazabilidad
| Tabla | Qué guarda | Por qué |
|---|---|---|
| `observacion` | Anomalías detectadas: informes sin estudio, grabaciones incompletas, FC imposibles, pacientes con FA en otro Holter, etc. | Que ningún problema de los datos quede oculto; cada uno se revisa y se marca como resuelto. |
| `almacen` | Las carpetas donde están las señales y los PDF. | Si se mueven las carpetas, se actualiza una sola fila. |
| `carga` | De qué archivo original salió cada dato (con su huella digital). | Reproducibilidad. |
| `migracion` | Cada corrección aplicada a los datos. | Historial de cambios auditable. |
| `respaldo` | Cada copia de seguridad hecha en el disco externo. | Saber si hay cambios sin respaldar. |

Además, las tablas `dim_*`, `fact_estudio` y `bridge_diagnostico` son una copia reorganizada de
estos mismos datos, pensada para hacer análisis estadísticos. Se regeneran solas a partir de las
tablas de arriba y no contienen nombres ni DNI.
