# install.packages("forestplot")
# install.packages("fpShapesGp")
library("forestplot")

data<-readr::read_csv(paste0("./","./sample.rr.csv"))
attach(data)
styles <- fpShapesGp(
  lines = list(
    gpar(col = "black"),
    gpar(col = "black"),
    gpar(col = "#3951a2"),
    gpar(col = "#5c90c2"),
    gpar(col = "#92c5de"),
    gpar(col = "#fdb96b"),
    gpar(col = "#f67948"),
    gpar(col = "#da382a"),
    gpar(col = "#a80326"),
    gpar(col = "#cccccc"),
    gpar(col = '#999999'),
    gpar(col = "#333333")
  ),
  box = list(
    gpar(fill = "black"),
    gpar(fill = "black"),
    gpar(fill = "#3951a2"),
    gpar(fill = "#5c90c2"),
    gpar(fill = "#92c5de"),
    gpar(fill = "#fdb96b"),
    gpar(fill = "#f67948"),
    gpar(fill = "#da382a"),
    gpar(fill = "#a80326"),
    gpar(fill = "#cccccc"),
    gpar(fill = '#999999'),
    gpar(fill = "#333333")
  ) 
)
forest<-forestplot(as.matrix(data[,1:3]), RR, Low, High, graph.pos=2, boxsize=0.3, zero=1, lineheight='auto', shapes_gp = styles)
png('forest_iv.png', res=300, units="cm",width=16,height=8)
plot(forest)
dev.off()

