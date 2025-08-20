EPOCHS = 20  # เทรน 20 รอบ (สามารถเพิ่มได้)
BATCH_SIZE = 64

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
