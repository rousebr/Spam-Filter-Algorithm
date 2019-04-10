#first run rpart with cp=0 because we dont know what cp to use
#second prune tree with correct cp based on 1-SE rule and plot

#rpart with unimportant variables removed is a bigger tree but with lower misclassification
#rpart without unimportant variables removed is smaller with higher misclassification
getwd()

set.seed(2)
spambase <- read.csv("~/Desktop/Stat 702/spambase/spambase.csv")
names(spambase)=c("word_freq_make","word_freq_address","word_freq_all","word_freq_3d","word_freq_our","word_freq_over",
    "word_freq_remove","word_freq_internet","word_freq_order",
    "word_freq_mail","word_freq_receive","word_freq_will","word_freq_people","word_freq_report","word_freq_addresses","word_freq_free","word_freq_business",
    "word_freq_email","word_freq_you","word_freq_credit","word_freq_your","word_freq_font","word_freq_000","word_freq_money","word_freq_hp","word_freq_hpl",
    "word_freq_george","word_freq_650","word_freq_lab","word_freq_labs","word_freq_telnet","word_freq_857","word_freq_data","word_freq_415","word_freq_85",
    "word_freq_technology","word_freq_1999","word_freq_parts","word_freq_pm","word_freq_direct","word_freq_cs","word_freq_meeting","word_freq_original",
    "word_freq_project","word_freq_re","word_freq_edu","word_freq_table","word_freq_conference","char_freq_;","char_freq_(","char_freq_[","char_freq_!",
    "char_freq_$","char_freq_#","capital_run_length_average","capital_run_length_longest","capital_run_length_total", "spam_nonspam")
attach(spambase)
library(rpart)
library(caret)
library(partykit)

#change to worded instead of binary 0,1
spambase$spam_nonspam = factor(spambase$spam_nonspam, levels=0:1, labels=c("email", "spam"))
table(spambase$spam_nonspam)

#build biggest tree
control= rpart.control(cp=0, xval = 10)
spamfit= rpart(spam_nonspam~., data = spambase, method="class", control = control)

#full tree display
plot(spamfit, uniform = T, compress = T)
text(spamfit, use.n=T)

#cp table and plot
printcp(spamfit) #.0018 cp using 1-se rule
plotcp(spamfit)

#splits vs cp
nsplits= spamfit$cptable[,2]
train_error= spamfit$cptable[,3]
xerror= spamfit$cptable[,4]
xstd= spamfit$cptable[,5]
plot(nsplits, train_error, type = 'l', xlab = "Number of splits", ylab = "Training error")
lines(nsplits, xerror, lty=3)
lines(nsplits, xerror+xstd, lty=4)
lines(nsplits, xerror-xstd, lty=4)
legend(17,1, c("train error", "xerror", "+/- 1 xstd"), lty = c(1,3,4))

#optimal pruned tree
spamfitprune= prune.rpart(spamfit, cp=0.0018)
print(spamfitprune)
plot(spamfitprune, uniform = T) #34 terminal nodes
text(spamfitprune, use.n = T)
summary(spamfitprune)

#8 terminal node tree
tree8 <- prune(spamfitprune, (spamfitprune$cptable[6,1] + spamfitprune$cptable[7,1])/2)
plot(as.party.rpart(tree8), uniform=T, margin=0.3)

#misclassification rate
pred_val <- predict(spamfitprune, newdata= spambase, type = "class")
table(spambase$spam_nonspam, pred_val)
confusionMatrix(spambase$spam_nonspam, pred_val)

##False Negative: message that is spam, but is incorrectly seen as an email 
##False Positive: message that is an email, but is incorrectly seen as a spam(1-specificity)

##################################part 2 with weighted loss matrix################
lmat <- matrix(c(0,10,1,0), nrow=2, byrow=T)
control= rpart.control(cp=0, xval = 10)
weighted_spamfit= rpart(spam_nonspam~., data = spambase, control = control, parms = list(loss=lmat), method = "class")

#full tree display
plot(weighted_spamfit, uniform = T, compress = T)
text(weighted_spamfit, use.n = T)

#cp table and plot
printcp(weighted_spamfit) #.0048 cp
plotcp(weighted_spamfit)

#splits vs cp
wnsplits= weighted_spamfit$cptable[,2]
wtrain_error= weighted_spamfit$cptable[,3]
wxerror= weighted_spamfit$cptable[,4]
wxstd= weighted_spamfit$cptable[,5]
plot(wnsplits, wtrain_error, type = 'l', xlab = "Number of splits", ylab = "Training error", ylim = c(range(wtrain_error)[1],range(wxerror)[2]))
lines(wnsplits, wxerror, lty=3)
lines(wnsplits, wxerror+wxstd, lty=4)
lines(wnsplits, wxerror-wxstd, lty=4)
legend(30,9, c("train error", "xerror", "+/- 1 xstd"), lty = c(1,3,4))

#optimal pruned tree
weighted_spamfitprune= prune.rpart(weighted_spamfit, cp=0.0048)
print(weighted_spamfitprune)
plot(weighted_spamfitprune, uniform = T, compress = T) #29 terminal nodes
text(weighted_spamfitprune, use.n = T)
summary(weighted_spamfitprune)

#8 terminal node tree
tree8_w <- prune(weighted_spamfitprune, (weighted_spamfitprune$cptable[5,1] + weighted_spamfitprune$cptable[6,1])/2)
plot(as.party(tree8_w), uniform=T, margin=0.3)


#misclassification rate
pred_valw <- predict(weighted_spamfitprune, newdata= spambase, type = "class")
table(spambase$spam_nonspam, pred_valw)
confusionMatrix(spambase$spam_nonspam, pred_valw)
