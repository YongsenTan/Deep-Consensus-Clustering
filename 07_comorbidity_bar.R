# install.packages('ggplot2')
library(ggplot2)
data<-readr::read_csv(paste0("./sample/com_bar.csv"))
group <- factor(data$com,levels=unique(data$com),order=TRUE)
png('com_bar_iv.png', res=300, width = 2400, height = 1600)
bar <- ggplot(data=data,aes(x=com,y=cnt,fill=S)) +
  geom_bar(stat="identity",position="fill") +
  scale_fill_manual(values=c(S1 = '#3951a2', S2 = "#5c90c2", S3 = "#92c5de", S4 = "#fdb96b", S5 = "#f67948", S6 = "#da382a", S7="#a80326"))+
  scale_y_continuous(expand  = expansion(mult=c(0.01,0.02)),
                     labels = scales::percent_format())+
  labs(x="",y="Relative Proportion",
       fill=" ",title="")+
  theme_bw()+
  theme(axis.title.y=element_text(size=14))+
  theme(legend.text=element_text(size=10))+
  theme(axis.text.x = element_text(size = 12, color = "black"))+
  theme(axis.text.y = element_text(size = 12, color = "black"))+
  theme(axis.ticks.length=unit(0.3,"cm"))+
  theme(axis.text.x=element_text(angle=45,vjust=1,hjust=1,size=11))

plot(bar)
# ggsave("kdigo_total_iv.png", dpi = 300, limitsize = FALSE)
dev.off()