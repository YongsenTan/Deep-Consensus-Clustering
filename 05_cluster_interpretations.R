# install.packages("ggalluvial")
library("ggalluvial")

png('cluster_interpret_iv.png', res=300, width = 5000, height = 2400)
data<-readr::read_csv(paste0("./sample/cluster_interpret.csv"))
levels(data$cluster) <- rev(levels(data$cluster))
res <- ggplot(data,
aes(x = c, stratum = cluster, alluvium = id,
   y = freq,
   fill = cluster, label = cluster)) +
    scale_fill_manual(values = c(S1 = '#3951a2', S2 = '#5c90c2', S3 = '#92c5de',S4 = '#fdb96b', S5 = '#f67948', S6 = '#da382a', S7 = '#a80326')) +
scale_x_continuous(name = 'Number of clusters', breaks = 1:9, labels = 1:9) +
  scale_y_continuous(name = 'Proportion of patients') +
geom_flow() +
geom_stratum(alpha = .5) +
geom_text(stat = "stratum", size = 3) +
theme(legend.position = "none")

plot(res)
dev.off()