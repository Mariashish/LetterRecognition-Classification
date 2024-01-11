# Carico i dati
load("Dati/DatiPresentazione2024.RData")

# Rimuovo tutti i dataset restanti che non utilizzo
remove(bfi, bfi.keys, Covid19Regioni, DatiRieti, Economics, Stirpat, StockReturns, wiki4HE)

# Seleziono il dataset 'letter'
data <- letter
remove(letter)

# Installa e carica le librerie necessarie
library(randomForest)  # Libreria per Random Forest
library(caret)         # Libreria per la gestione dei dati e delle predizioni
library(ggplot2)       # Libreria per la creazione di grafici
library(e1071)         # Libreria per Support Vector Machine (SVM)
library(class)         # Libreria per k-Nearest Neighbor (k-NN)

# Assegna i nomi alle colonne
colnames(data) <- c("lettr", "x_box", "y_box", "width", "high", "onpix", "x_bar", "y_bar", "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x_ege", "xegvy", "y_ege", "yegvx")

# Converti 'lettr' in un fattore
data$lettr <- as.factor(data$lettr)

# Split del dataset in training e testing sets
set.seed(123)  # per riproducibilità
train_indices <- sample(1:nrow(data), 0.75 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Train di un modello Random Forest
rf_model <- randomForest(lettr ~ ., data = train_data)

# Predizioni sul test set
predictions_rf <- predict(rf_model, test_data)

# Valutazione dell'accuratezza del modello RF
accuracy_rf <- sum(predictions_rf == test_data$lettr) / nrow(test_data)
cat("Random Forest Accuracy:", accuracy_rf, "\n")

# Matrice di confusione per il modello RF
confusion_matrix_rf <- confusionMatrix(predictions_rf, test_data$lettr)
print(confusion_matrix_rf)

# Train di un modello Support Vector Machine (SVM)
svm_model <- svm(lettr ~ ., data = train_data, kernel = "linear")

# Predizioni sul test set per il modello SVM
svm_predictions <- predict(svm_model, test_data)

# Valutazione dell'accuratezza del modello SVM
accuracy_svm <- sum(svm_predictions == test_data$lettr) / nrow(test_data)
cat("SVM Accuracy:", accuracy_svm, "\n")

# Matrice di confusione per il modello SVM
confusion_matrix_svm <- confusionMatrix(svm_predictions, test_data$lettr)
print(confusion_matrix_svm)

# Train di un modello k-NN
knn_model <- knn(train_data[, -1], test_data[, -1], train_data$lettr, k = 5)

# Valutazione dell'accuratezza del modello k-NN
accuracy_knn <- sum(knn_model == test_data$lettr) / nrow(test_data)
cat("k-NN Accuracy:", accuracy_knn, "\n")

# Matrice di confusione per il modello k-NN
confusion_matrix_knn <- confusionMatrix(knn_model, test_data$lettr)
print(confusion_matrix_knn)

# Visualizzazione grafica delle accuracies
accuracy_data <- data.frame(Model = c("Random Forest", "SVM", "k-NN"),
                            Accuracy = c(accuracy_rf, accuracy_svm, accuracy_knn))

# Bar plot delle accuracies con scala y dettagliata ogni 5
ggplot(accuracy_data, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Confronto delle Accuratezze dei Modelli", x = "Modello", y = "Accuratezza") +
  theme_minimal() +
  scale_y_continuous(breaks = seq(0, 1, by = 0.05))
