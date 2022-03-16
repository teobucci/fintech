#Holy Bible

#############Preliminar##################################################
# Recall: qqplot to verify (qualitatively) the Gaussian assumption on the 
# distribution generating sample (m100 è una variabile!)
qqnorm(m100) # quantile-quantile plot
qqline(m100, col='red') # theoretical line
# Recall: Shapiro-Wilk test to verify (quantitatively) the Gaussian assumption on the
# distribution generating sample
shapiro.test(m100)

# Comandi base regression
regression <- lm(m200 ~ m100)
regression

summary(regression)

coef(regression)
vcov(regression)
residuals(regression)
fitted(regression)

points(m100, fitted(regression))
#plotta i punti della regressione sopra lo scatterplot

# Confidence and prediction intervals (command predict)
newdata <- data.frame(m100=c(10,11,12))
pred_nd <- predict(regression, newdata)

IC_nd <- predict(regression, newdata, interval = 'confidence', level = .99)
IP_nd <- predict(regression, newdata, interval = 'prediction', level = .99)

#fa tutti i grafici da se sgravatissimo
x11()
par (mfrow=c(2,2))
plot(regression)

aneurysm[,5] <- factor(aneurysm[,5]) #rende la colonna un factor

#############Graphs######################################################

# Scatter plot e istogrammi
plot(m100,m200)  #m100 e m200 sono colonne
hist(m100, prob=T)
hist(m200, prob=T)

pairs(record) #record è un dataset

#Boxplot
boxplot(tourists, las=2, col='gold') #tourists è il dataset
boxplot(scale(x=tourists,center = T, scale=F), las=2, col='gold')
#scale scala i dati, scelgo dove centrarli,scale f vuol dire che li traslo e basta
#scale t invece divide anche per la standard deviation

# matplot
matplot(t(aneurysm.geometry),type='l')
matplot(t(aneurysm.geometry),type='l',col=color.position)
#Plot the columns of one matrix against the columns of another 
# (which often is just a vector treated as 1-column matrix).

# Pie chart (no ordering of levels) (solo per categoriche)
x11()
pie(table(district),col=rainbow(length(levels(district))))

# Barplot (levels are ordered) (solo per categoriche)
x11()
barplot(table(district)/length(district))
#istogramma delle frequenze realtive
plot(district)   # barplot of absolute frequences

############PCA, verifica della teoria#####################################
#X è il dataset
M <- colMeans(X)   #mean di una matrice , quindi un vettore
S <- cov(X)   #covariance matrix
# we compute the eigenvectors and eigenvalues
eigen(S)
# Note. eigen(S)$vectors returns a matrix whose 
#     columns are the eigenvectors of S
# eigen(S)$values mi da gli autovalori

library(car)
ellipse(M, S, 1, add=T,lwd=3, col='red')

#grafici
abline(a = M[2] - eigen(S)$vectors[2,1]/eigen(S)$vectors[1,1]*M[1], b = eigen(S)$vectors[2,1]/eigen(S)$vectors[1,1], lty = 2, col = 'dark red', lwd = 2)
abline(a = M[2] - eigen(S)$vectors[2,2]/eigen(S)$vectors[1,2]*M[1], b = eigen(S)$vectors[2,2]/eigen(S)$vectors[1,2], lty = 2, col = 'red', lwd = 2)

################PCA vera##################################################
pc.tourists <- princomp(tourists, scores=T)
summary(pc.tourists)

# To obtain the rows of the summary:
# standard deviation of the components
pc.tourists$sd
# proportion of variance explained by each PC
pc.tourists$sd^2/sum(pc.tourists$sd^2)
# cumulative proportion of explained variance
cumsum(pc.tourists$sd^2)/sum(pc.tourists$sd^2)  #ovviamente l'ultimo fara sempre 1

load.tour <- pc.tourists$loadings #estrae i loading:i loading sono i coefficienti applicati 
#alle variabili originarie per determinare le componenti principali.

# graphical representation of the loadings of the first 8 principal components
x11()
par(mfcol = c(4,2))
for(i in 1:8) barplot(load.tour[,i], ylim = c(-1, 1), main=paste("PC",i))

# We compute the standardized variables
tourists.sd <- scale(tourists)
tourists.sd <- data.frame(tourists.sd) #creo nuovo dataset standardizzato

# scores
scores.tourists <- pc.tourists$scores
x11()
plot(scores.tourists[,1:2])
abline(h=0, v=0, lty=2, col='grey')
#come i dati si distribuiscono rispetto alle 2 principal component

biplot(pc.tourists) # instead of plotting points, plotto il numero della riga del dato


