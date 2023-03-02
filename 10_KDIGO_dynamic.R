# install.packages("ggalluvial")
library("ggalluvial")

png('kdigo_dynamic.png', res=300, width = 2400, height = 1200)
data<-readr::read_csv(paste0("./sample/KDIGO_dynamic.csv"))
levels(data$data$aki_stage_max) <- rev(levels(data$data$aki_stage_max))
res <- ggplot(data,
aes(x = daydiff, stratum = data$aki_stage_max, alluvium = id,
   y = freq,
   fill = aki_stage_max, label = aki_stage_max)) +
scale_x_continuous(name = 'Time after admission (hours)', breaks = 0:2, labels = c("24", "48", "72")) +
  scale_y_continuous(name = 'Number of patients') +
geom_flow() +
geom_stratum(alpha = .5) +
geom_text(stat = "stratum", size = 3) +
theme(legend.position = "none")

plot(res)
dev.off()