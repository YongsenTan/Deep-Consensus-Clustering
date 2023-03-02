library("survminer")
require("survival")

#loading data
Data<-readr::read_csv(paste0("./sample/survival.csv"))

fit<- survfit(Surv(los, y) ~ sub, data = Data)
png('survival_iv.png', res=300, width = 2400, height = 1800)
surp<-ggsurvplot(fit,
                 # title = "AKI patients in MIMIC-IV",
                 legend.title = "Subphenotypes",
                 legend.labs = c("S1", "S2", "S3", "S4", "S5", "S6", "S7", "Stage 1", "Stage 2", "Stage 3"),
                 palette = c('#3951a2','#5c90c2','#92c5de', '#fdb96b','#f67948', '#da382a', '#a80326', "#cccccc", '#999999', "#333333"),
                 size = 0.5,
                 censor = FALSE, # True

                 # font.main = 10,
                 # font.x = 5,
                 # font.y = 5,
                 # font.tickslab = 5,
                 conf.int = TRUE, # Add confidence interval
                 pval = TRUE, # Add p-value
                 #pval.method = TRUE,
                 surv.plot.height = 0.7,
                 
                 risk.table = TRUE,        # Add risk table
                 risk.table.col = "strata",# Risk table color by groups
                 risk.table.height = 0.3, # Useful to change when you have multiple groups
                 risk.table.y.text.col = T, # colour risk table text annotations.
                 risk.table.y.text = FALSE, # show bars instead of names in text annotations

                 risk.table.fontsize = 3,
                 xlab = 'Time (Day)',
                 #xscale = 30,
                 #xscale = "d_m",
                 xlim = c(0, 14),
                 ylim = c(0, 1),
                 break.time.by = 1,
                 ggtheme = theme_light(),
                 surv.median.line = "hv",  # add the median survival pointer.
)
surp

grid.draw.ggsurvplot <- function(x){
survminer:::print.ggsurvplot(x, newpage=FALSE)
}

dev.off()

