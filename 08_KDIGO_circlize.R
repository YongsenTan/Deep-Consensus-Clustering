# install.packages('circlize')
library(circlize)

mat <-readr::read_csv("./sample/KDIGO_circlize.csv")
rownames(mat) <- paste0("Stage", 1:3)
colnames(mat) <- paste0("S", 1:7)

grid.col <- c(Stage1 = "grey", Stage2 = "grey", Stage3 = "grey",
    S1 = '#3951a2', S2 = "#5c90c2", S3 = "#92c5de", S4 = "#fdb96b", S5 = "#f67948", S6 = "#da382a", S7="#a80326")
order <- c("S7", "S6", "S5", "S4", "S3", "S2", "S1", "Stage1", "Stage2", "Stage3")
png('KDIGO_circlize.png', res=300, width = 2400, height = 2400)
chordDiagram(t(mat), grid.col = grid.col)
dev.off()
