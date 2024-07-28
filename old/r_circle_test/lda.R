library(circlepackeR)
library(data.tree)
library(treemap)
library("rjson")

data <- fromJSON(file = "C:\\Users\\shive\\Desktop\\llnl-ksu-recipes\\tkm-repo\\lda_data.json")

circlepackeR(data, color_max = "hsl(341,30%,40%)")