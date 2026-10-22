# Downstream frequency scoring and statistical analyses (original SLT script)
# -----------------------------------------------------------------------------
# Python extraction outputs rule occurrences but not corpus-based frequencies.
# This script merges those occurrences with precomputed Switchboard lookup
# tables, aggregates sentence-level frequency measures, and runs the analyses
# and plots in the original "A Syntax-Lexicon Trade-off in Language Production"
# workflow. This is provided for provenance and as a reference for feature
# calculation; the title/figure numbering may not match the JPAD manuscript.
#
# Expected working directory: repository root (the folder containing Data/).
# Required data, participant metadata and lookup dictionaries are listed in
# README.md. No analytical model formulas have been modified.
# -----------------------------------------------------------------------------

library(tidyverse)
library(lme4)
library(lmerTest)
library(readxl)
library(Hmisc)
library(caret)


################
# 0. Attach reference-corpus log frequencies to extracted words and rules.
# The counts are generated in Python; frequencies are assigned here via
# precomputed Switchboard dictionaries (log[count + 1] per type).
word.rules = bind_rows(read_csv("Data/wordsrules_content.csv"),
                       read_csv("Data/wordsrules_nocontent.csv") )

# Optional provenance: code to build the token frequency lookup from the
# full B-prefixed Switchboard corpus is retained here but commented out.
# switchboard large for word freqs
# switchboard.token = filter(word.rules, grepl("^B", SubjectID)) %>%
#   group_by(ContentWordToken) %>%
#   summarise(WordTokenFreq=log(n() + 1),
#             Content.Pct = mean(IsContentWord),
#             n=n()) %>%
#   arrange(-WordTokenFreq)
# write_csv(switchboard.token, file = "Data/SwitchboardDicts/switchboard_token.csv")
switchboard.token = read_csv("Data/SwitchboardDicts/switchboard_token_large.csv")

# Calculate sentence-level means for the small Switchboard reference sample.
# Switchboard cleaned for syntax analysis
s.word = filter(word.rules, grepl("ss", SubjectID)) %>%
  left_join(switchboard.token) %>%
  mutate(ContentFreq = ifelse(IsContentWord == TRUE, WordTokenFreq, NA)) %>%
  group_by(SentenceID) %>%
  summarise(WordTokenFreq=mean(WordTokenFreq, na.rm=T),
            ContentFreq=mean(ContentFreq, na.rm=T))


# Restrict target-cohort token rows; unseen tokens receive frequency 0.
word.rules = filter(word.rules, grepl("[JPS]", SubjectID),
                    !grepl("Switchboard", SubjectID))

word.rules = left_join(word.rules, select(switchboard.token, -Content.Pct)) %>%
  mutate(WordTokenFreq = ifelse(is.na(WordTokenFreq),
                                            0,
                                            WordTokenFreq),
         ContentFreq = ifelse(IsContentWord == TRUE, WordTokenFreq, NA))


# Each row is a dependency-derived syntax-rule occurrence, not yet a
# reference-corpus frequency.
syntax.rules = read.csv("Data/syntaxrules.csv")
# Optional provenance: construct lookup by counting rule occurrences in
# B-prefixed Switchboard rows; the existing CSV is normally loaded instead.
# switchboard.syntax = filter(syntax.rules, grepl("^B", SubjectID)) %>%
#   group_by(SyntaxRule_original) %>%
#   summarise(SyntaxFrequency_original=log(n() + 1)) %>%
#   arrange(-SyntaxFrequency_original)
# write_csv(switchboard.syntax, "Data/SwitchboardDicts/switchboard_syntax_large.csv")
switchboard.syntax = read_csv("Data/SwitchboardDicts/switchboard_syntax_large.csv")

s.syntax = filter(syntax.rules, grepl("^ss", SubjectID)) %>%
  left_join(switchboard.syntax) %>%
  group_by(SentenceID) %>%
  summarise(SyntaxFreq=mean(SyntaxFrequency_original, na.rm=T))


syntax.rules = filter(syntax.rules, grepl("[JPS]", SubjectID))

# Unknown rules are assigned 0 after joining to the reference dictionary.
syntax.rules = syntax.rules %>%
  left_join(switchboard.syntax) %>%
  mutate(SyntaxFrequency_original = ifelse(is.na(SyntaxFrequency_original),
                0,
                SyntaxFrequency_original))


#_______________________________________________________
# 1.1 Study 1: word frequency by PPA subtype (mixed-effects models).
# SubjectID is the random intercept; utterance length is a covariate.
study1info = read_csv("Data/Study1Info.csv")
d.study1.utterance = left_join(study1info, read_csv("Data/alternative_metrics.csv")) 
d.study1.utterance$SentenceLength = d.study1.utterance$length
# Each sentence receives the mean log-frequency of its extracted rules.
d.study1.syntax = filter(syntax.rules, grepl("P", SubjectID)) %>%
  group_by(SentenceID) %>%
  dplyr::summarize(SyntaxFrequency=mean(SyntaxFrequency_original))
# Calculate sentence means for all tokens and for content words only.
d.study1.words = filter(word.rules, grepl("P", SubjectID)) %>%
  group_by(SentenceID) %>%
  summarise(WordTokenFreq = mean(WordTokenFreq),
            ContentFreq = mean(ContentFreq, na.rm=T))
d = left_join(d.study1.utterance, d.study1.syntax) %>%
  left_join(d.study1.words)


m1 = lmer(data=d, ContentFreq ~ Subtype + SentenceLength +
            (1  |SubjectID),
          control = lmerControl(optimizer ="Nelder_Mead"))
summary(m1)

m2 = lmer(data=d, ContentFreq ~ SentenceLength + (1  |SubjectID),
          control = lmerControl(optimizer ="Nelder_Mead"))
summary(m2)

# Nested-model comparison: contribution of PPA subtype.
anova(m1,m2) 

#footnote
summary(lmer(data=d, WordTokenFreq ~ Subtype + SentenceLength +
            (1 |SubjectID)),
        control = lmerControl(optimizer ="Nelder_Mead"))

#_______________________________________________________
# 1.2.1 Compare syntax frequency between PPA variants and controls.
m3 = lmer(data=d, SyntaxFrequency ~ Subtype + SentenceLength + (1|SubjectID))
summary(m3)

m4 = lmer(data=d, SyntaxFrequency ~ SentenceLength + (1|SubjectID))
summary(m4)

anova(m3,m4)

# 1.2.2 Repeat after matching the group-by-utterance-length distribution.
# Sampling is reproducible because the random seed is fixed below.
min.n = group_by(d, length, Subtype) %>%
  summarise(n=n()) %>%
  right_join(expand.grid(length=1:max(d$length, na.rm=t), Subtype=c("lvPPA", "control", "nfvPPA", "svPPA"))) %>%
  group_by(length) %>%
  summarise(minn=min(n)) %>%
  mutate(minn = ifelse(is.na(minn), 0, minn))

set.seed(42)
d.matched = group_by(d, Subtype, length) %>%
  mutate(sample_num = sample(1:n())) %>%
  left_join(min.n) %>%
  filter(sample_num <= minn)

table(d.matched$Subtype, d.matched$length)
m3.1 = lmer(data=d.matched, SyntaxFrequency ~ Subtype + SentenceLength + (1|SubjectID))
summary(m3.1)

m4.1 = lmer(data=d.matched, SyntaxFrequency ~ SentenceLength + (1|SubjectID))
summary(m4.1)

anova(m3.1,m4.1)

#_______________________________________________________
# 1.3 Syntax-lexicon trade-off: test the within/between-utterance relation
# using subject-level mixed effects and subject-level correlations.

# maximal mixed effect
m5.content = summary(lmer(ContentFreq ~ SyntaxFrequency + SentenceLength + (1 + SyntaxFrequency ||SubjectID), data=d))
summary(m5.content) 

#footnote
m5.all = summary(lmer(WordTokenFreq ~ SyntaxFrequency + SentenceLength + (1 + SyntaxFrequency ||SubjectID), data=d))
summary(m5.all) 

m5.subtype = summary(lmer(ContentFreq ~ SyntaxFrequency + SentenceLength + (1 + SyntaxFrequency ||SubjectID) +
                            (1 |Subtype), data=d))
summary(m5.subtype) 

# Collapse per-utterance values to one record per participant.
d.subject.level = group_by(d, SubjectID, Subtype) %>%
  summarise(SyntaxFrequency = mean(SyntaxFrequency, na.rm=T),
            SentenceLength = mean(SentenceLength, na.rm=T),
            ContentFreq = mean(ContentFreq, na.rm=T),
            WordTokenFreq = mean(WordTokenFreq, na.rm=T),
            MaxDep = mean(maxdep, na.rm=T),
            Dist = mean(avg_dist, na.rm=T),
            VerbRatio = mean(noun_verb_ratio, na.rm=T))
pos.ent = read_csv("Data/pos_entropy.csv")
d.subject.level = left_join(d.subject.level, pos.ent)

#subject level correlation
rcorr(d.subject.level$ContentFreq,
      d.subject.level$SyntaxFrequency, type = 'pearson') 

rcorr(d.subject.level$WordTokenFreq,
      d.subject.level$SyntaxFrequency, type = 'pearson') 

#_______________________________________________________
# 1.4 Binary classification of nfvPPA versus other diagnostic groups.
# Leave-one-subject-out cross-validation compares candidate predictors.
# Nagelkerke pseudo-R^2 is computed on full-data logistic fits.

nums = NULL

get.Nagel = function(m, md) {
  CoxSnellNagelkerke<- 1 - exp ((m$deviance-m$null.deviance)/nrow(md) ) #this is another measure of R2
  Nagelkerke<- CoxSnellNagelkerke/(1-(exp(-(m$null.deviance/nrow(md) ))))
  return(Nagelkerke)
}

# Predictor sets are: feature alone; +sentence length; +content frequency.
for (variable in c("SyntaxFrequency", "SentenceLength", "WordTokenFreq", 
            "MaxDep", "Dist", "VerbRatio",'POSentropy')) {
  mdata1<-d.subject.level
  mdata1$nfvPPA = ifelse(mdata1$Subtype == "nfvPPA", 1, 0)
  mdata1$variable = mdata1[, variable][[1]]
  
  acc = NULL
  acc2 = NULL
  acc3 = NULL
  for (i in 1:nrow(mdata1))
  {
    DataTest = mdata1[i, ]
    DataTrain = mdata1[-i, ]
    m1 = glm(nfvPPA~variable, data=DataTrain, family='binomial')
    m2 = glm(nfvPPA~variable + SentenceLength, data=DataTrain, family='binomial')
    m3 = glm(nfvPPA~variable + SentenceLength + ContentFreq, data=DataTrain, family='binomial')
    acc = append(acc, mean(round(predict(m1,newdata=DataTest, type='response')) == DataTest$nfvPPA))
    acc2 = append(acc2, mean(round(predict(m2,newdata=DataTest, type='response')) == DataTest$nfvPPA))
    acc3 = append(acc3, mean(round(predict(m3,newdata=DataTest, type='response')) == DataTest$nfvPPA))
    
  }
  m1 = glm(nfvPPA~variable, data=mdata1, family='binomial')
  m2 = glm(nfvPPA~variable + SentenceLength, data=mdata1, family='binomial')
  m3 = glm(nfvPPA~variable + SentenceLength + ContentFreq, data=mdata1, family='binomial')
  nums = rbind(nums, cbind(variable,
                           as.numeric(mean(acc)),
                           as.numeric(get.Nagel(m1, mdata1)),
                           as.numeric(mean(acc2)),
                           as.numeric(get.Nagel(m2, mdata1)),
                           as.numeric(mean(acc3)),
                           as.numeric(get.Nagel(m3, mdata1))
                           ))
}
nums = data.frame(nums)
names(nums) = c("variable", "acc.simple", "R2.simple", "acc.length", "R2.length", "acc.length.word", "R2.length.word")
nums[, 2:7] <- sapply(nums[, 2:7], as.numeric, round, 2)
nums[, 2:7] <- sapply(nums[, 2:7], round, 2)
write_csv(nums, "Data/table2.csv")


#_______________________________________________________
# 2.1 Healthy participants describing the picnic picture:
# sentence-level and subject-level syntax/lexicon associations.

study2info = read_csv("Data/Study2Info.csv")
d.study2.utterance = left_join(study2info, read_csv("Data/alternative_metrics.csv")) 
d.study2.utterance$SentenceLength = d.study2.utterance$length
d.study2.syntax = filter(syntax.rules, grepl("^S", SubjectID))  %>%
  group_by(SentenceID) %>%
  dplyr::summarize(SyntaxFrequency=mean(SyntaxFrequency_original, na.rm=T))
d.study2.words = filter(word.rules, grepl("^S", SubjectID))  %>%
  group_by(SentenceID) %>%
  summarise(WordTokenFreq = mean(WordTokenFreq),
            ContentFreq = mean(ContentFreq, na.rm=T))

d2 = left_join(d.study2.utterance, d.study2.syntax) %>%
  left_join(d.study2.words)

m6 = lmer(ContentFreq ~ SyntaxFrequency + SentenceLength + (1 + SyntaxFrequency  |SubjectID), data=d2)
summary(m6) 

#footnote
summary(lmer(WordTokenFreq ~ SyntaxFrequency + SentenceLength + (1 + SyntaxFrequency  |SubjectID), data=d2))


d2.subject.level = group_by(d2, SubjectID) %>%
  summarise(SyntaxFrequency = mean(SyntaxFrequency, na.rm=T),
            SentenceLength = mean(SentenceLength),
            WordTokenFreq = mean(WordTokenFreq),
            ContentFreq = mean(ContentFreq,na.rm=T))
rcorr(d2.subject.level$ContentFreq,
      d2.subject.level$SyntaxFrequency, type = 'pearson') 

#footnote
rcorr(d2.subject.level$WordTokenFreq,
      d2.subject.level$SyntaxFrequency, type = 'pearson') 

# Test whether the trade-off slope differs between patient and control
# samples, first at utterance level and then at participant level.
d2$Subtype = "control"
d.combined = bind_rows(d, d2)
d.combined$Patient = ifelse(d.combined$Subtype == "control", 0, 1)
m7<-lmer(ContentFreq ~ SyntaxFrequency + Patient +
           SentenceLength + SyntaxFrequency : Patient+ (1|SubjectID), data= d.combined)
summary(m7)

d2.subject.level$Subtype = "control"
d.combined.subj = bind_rows(d.subject.level, d2.subject.level)
d.combined.subj$Control = ifelse(d.combined.subj$Subtype == "control", 1, 0)
m8<-lm(ContentFreq ~ SyntaxFrequency + Control +
         scale(SentenceLength) + SyntaxFrequency : Control, data= d.combined.subj)
summary(m8)

d.combined.subj$Patient = ifelse(d.combined.subj$Subtype != "control", 1, 0)
# NOTE: SenteneLength below is an unresolved typo in the source
# (likely SentenceLength); retained without changing the original model.
m9<-lm(ContentFreq ~ SyntaxFrequency + Patient +
         scale(SenteneLength) + SyntaxFrequency : Patient, d.combined.subj)
summary(m9)

# 2.2 Job-description task: deliberately excluded from the public release
# for participant privacy. The original commented code is preserved below.
######### 2.2 (commented out because we do not release the job description task, for privacy reasons)
# mturk
# d.study3.utterance = read_csv("Data/alternative_metrics.csv") %>%
#   filter(grepl("^J", SubjectID))
# d.study3.utterance$SentenceLength = d.study3.utterance$length
# d.study3.syntax = filter(syntax.rules, grepl("^J", SubjectID))  %>%
#   group_by(SentenceID) %>%
#   dplyr::summarize(SyntaxFrequency=mean(SyntaxFrequency_original, na.rm=T))
# d.study3.words = filter(word.rules, grepl("^J", SubjectID))  %>%
#   group_by(SentenceID) %>%
#   summarise(WordTokenFreq = mean(WordTokenFreq),
#             ContentFreq = mean(ContentFreq, na.rm=T))
# 
# d3 = left_join(d.study3.utterance, d.study3.syntax) %>%
#   left_join(d.study3.words)
# 
# mj.token = lmer(data=d3, WordTokenFreq ~ 
#                   (SyntaxFrequency) + (SentenceLength) + 
#                   (1 + SyntaxFrequency|SubjectID))
# summary(mj.token) 
# 
# mj.content = lmer(data=d3, ContentFreq ~ (SyntaxFrequency) +
#                     (SentenceLength) + (1 + SyntaxFrequency |SubjectID))
# summary(mj.content)


# 2.3 Analyze the small Switchboard reference sample independently.
######### 2.3
s = left_join(s.syntax, s.word)

am = read_csv("Data/alternative_metrics.csv")

s = left_join(s, am)


filter(s, is.na(ContentFreq) == F, is.na(SyntaxFreq) == F) %>%
  summarise(a = cor(ContentFreq, SyntaxFreq))

filter(s, is.na(WordTokenFreq) == F, is.na(SyntaxFreq) == F) %>%
  summarise(a = cor(WordTokenFreq, SyntaxFreq))


table(s$length)
summary(lm(data=s, WordTokenFreq ~ (SyntaxFreq) + (length) ))
summary(lm(data=s, ContentFreq ~ (SyntaxFreq) + (length) ))


#________________________________________________________
# Figures: ggplot code renders into the active R graphics device;
# no image files are automatically saved by these statements.
#Figures produced by R
#figure 3A
mg<-ggplot(d,aes(ContentFreq,colour = Subtype))
mg + geom_density(size = 1) + scale_color_manual(values=c("#050d99", '#015c19', "#cf0202", '#fc6d00')) +
  theme_classic() +
  xlab('\nWord frequency (log)') +
  ylab('Density') +
  theme(axis.title = element_text(size = 10)) +
  theme(axis.text.x = element_text(size = 13),axis.title=element_text(size=19)) +
  theme(axis.text.y = element_text(size = 13)) 

#figure 3B
mg<-ggplot(d,aes(SyntaxFrequency,colour = Subtype))
mg + geom_density(size = 1) + scale_color_manual(values=c("#050d99", '#015c19', "#cf0202", '#fc6d00')) +
  theme_classic() +
  xlab('\nSyntax frequency (log)') +
  ylab('Density') +
  theme(axis.title = element_text(size = 10)) +
  theme(axis.text.x = element_text(size = 13),axis.title=element_text(size=19)) +
  theme(axis.text.y = element_text(size = 13)) 

#figure 3C
ggplot(data = d.subject.level, aes(x = Subtype, y = ContentFreq)) +
  geom_boxplot(
    position = position_dodge(width=0.5),
    outlier.shape = 17,
    outlier.colour = 'red2',
    outlier.size = 0.1,
    aes(fill = Subtype)) +
  scale_fill_manual(values = c( '#f5f5f5', '#f5f5f5','#f5f5f5','#f5f5f5')) + # for boxplot
  #Add the scatter points (treats outliers same as 'inliers')
  geom_jitter(
    position = position_jitter(width = 0.17),
    size = 3.0,
    aes(colour = Subtype)) +
  scale_color_manual(values = c("#050d99", '#015c19', "#cf0202", "#fc6d00")) +
  theme_classic() +
  theme(axis.text.x = element_text(size = 15, color = 'black'),axis.title=element_text(size=10))

#figure 3D
# NOTE: SyntaxFrequency1 is absent from the data frame constructed above.
# Original plotting variable retained; verify against your final data.
ggplot(data = d.subject.level, aes(x = Subtype, y = SyntaxFrequency1)) +
  geom_boxplot(
    position = position_dodge(width=0.5),
    outlier.shape = 17,
    outlier.colour = 'red2',
    outlier.size = 0.1,
    aes(fill = Subtype)) +
  scale_fill_manual(values = c( '#f5f5f5', '#f5f5f5','#f5f5f5','#f5f5f5')) + # for boxplot
  #Add the scatter points (treats outliers same as 'inliers')
  geom_jitter(
    position = position_jitter(width = 0.17),
    size = 3.0,
    aes(colour = Subtype)) +
  scale_color_manual(values = c("#050d99", '#015c19', "#cf0202", "#fc6d00")) +
  theme_classic() +
  theme(axis.text.x = element_text(size = 15, color = 'black'),axis.title=element_text(size=10))

#figure 4A
mg<-ggplot(d,aes(SyntaxFrequency,ContentFreq,color=Subtype))
mg + geom_point() + scale_color_manual(values=c("#050d99", '#015c19', "#cf0202", '#fc6d00')) +
  theme_classic() +
  xlab('\nSyntax frequency (log)') +
  ylab('Word frequency (log)') +
  geom_smooth(method='lm',color = '#363131', alpha=0) +
  theme(axis.title = element_text(size = 15))

#figure 4B
# NOTE: SyntaxFrequency1 is absent from the data frame constructed above.
# Original plotting variable retained; verify against your final data.
mg = ggplot(d.subject.level,aes(SyntaxFrequency1,
                                WordTokenFreq,
                                color=Subtype))

mg + geom_point() + scale_color_manual(values=c("#050d99", '#015c19', "#cf0202", '#fc6d00')) +
  theme_classic() +
  xlab('\nSyntax frequency (log)') +
  ylab('Word frequency (log)') +
  geom_smooth(method='lm',color = '#363131', alpha=0) +
  theme(axis.title = element_text(size = 15))

#figure 4C
mg<-ggplot(d2,aes(SyntaxFrequency,ContentFreq))
mg + geom_point(color = '#5d00a8') +
  theme_classic() +
  xlab('\nSyntax frequency (log)') +
  ylab('Word frequency (log)') +
  geom_smooth(method='lm',color = '#363131', alpha=0) +
  theme(axis.title = element_text(size = 16))

#figure 4D
mg<-ggplot(d2.subject.level,aes(SyntaxFrequency,ContentFreq))
mg + geom_point(color = '#5d00a8') +
  theme_classic() +
  xlab('\nSyntax frequency (log)') +
  ylab('Word frequency (log)') +
  geom_smooth(method='lm',color = '#363131', alpha=0) +
  theme(axis.title = element_text(size = 15))

# Supplementary Table 2: leave-one-subject-out multinomial classification
# across PPA subtypes, reporting accuracy and full-model AIC.
#____________________Supplementary material 2
library(nnet)
nums = NULL

for (variable in c("SyntaxFrequency", "SentenceLength", "WordTokenFreq", 
                   "MaxDep", "Dist", "VerbRatio",'POSentropy')) {
  mdata1<-d.subject.level
  mdata1$variable = mdata1[, variable][[1]]
  mdata1 = filter(mdata1, is.na(variable) == F)
  acc = NULL
  acc2 = NULL
  acc3 = NULL
  for (i in 1:nrow(mdata1))
  {
    DataTest = mdata1[i, ]
    DataTrain = mdata1[-i, ]
    m1 = multinom(Subtype~variable, data=DataTrain) 
    m2 = multinom(Subtype~variable + SentenceLength, data=DataTrain)
    m3 = multinom(Subtype~variable + SentenceLength + ContentFreq, data=DataTrain)
    acc = append(acc, predict(m1,newdata=DataTest) == DataTest$Subtype)
    acc2 = append(acc2, predict(m2, newdata=DataTest) == DataTest$Subtype)
    acc3 = append(acc3, predict(m3,newdata=DataTest) == DataTest$Subtype)
  }
  m1 = multinom(Subtype~variable, data=mdata1) 
  m2 = multinom(Subtype~variable + SentenceLength, data=mdata1)
  m3 = multinom(Subtype~variable + SentenceLength + ContentFreq, data=mdata1)
  nums = rbind(nums, cbind(variable,
                           as.numeric(mean(acc)),
                           as.numeric(m1$AIC),
                           as.numeric(mean(acc2)),
                           as.numeric(m2$AIC),
                           as.numeric(mean(acc3)),
                           as.numeric(m3$AIC)))
}
nums = data.frame(nums)
# The original output headers say R2, but the values in those columns
# are AIC for these multinomial models. Labels are kept for compatibility.
names(nums) = c("variable", "acc.simple", "R2.simple", "acc.length", "R2.length", "acc.length.word", "R2.length.word")
nums[, 2:7] <- sapply(nums[, 2:7], as.numeric, round, 2)
nums[, 2:7] <- sapply(nums[, 2:7], round, 2)
write_csv(nums, "Data/tableS2.csv")


