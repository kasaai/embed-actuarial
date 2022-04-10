library(recipes)
library(rsample)
library(data.table)
library(dplyr)

dirs = "C:\\R\\embed-actuarial\\results\\"

files = list.files(dirs)

all_res = lapply(files, function (x) fread(paste0(dirs, x))) %>% rbindlist(fill = T) %>% data.table()

all_res = all_res %>% melt.data.table(id.vars = c("run", "id"))
melted = all_res[, mean(value), keyby = .(run, variable)]
melted = melted[order(run)]%>%
  dcast.data.table(variable~run)

melted[variable %in% c("rmse_nn", "rmse_nn2", "mae_nn", "mae_nn2")] %>%
  fwrite("c:/r/finetune_sect4.csv")

all_res = fread(paste0(dirs, "sect5.csv"))

all_res = all_res %>% melt.data.table(id.vars = c("run", "id"))
melted = all_res[, mean(value), keyby = .(run, variable)]
melted = melted[order(run)]%>%
  dcast.data.table(variable~run)

melted %>%
fwrite("c:/r/finetune_sect5.csv")




all_res = res

all_res = all_res %>% melt.data.table(id.vars = c("run", "id"))
melted = all_res[, mean(value), keyby = .(run, variable)]
melted = melted[order(run)]%>%
  dcast.data.table(variable~run)

melted %>%
  fwrite("c:/r/finetune_sect5.csv")
