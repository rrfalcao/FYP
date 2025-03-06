import tensorflow as tf

def dice_coef(y_true, y_pred): 
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-6) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-6) 

def dice_loss(y_true, y_pred): 
    return -dice_coef(y_true, y_pred)

def log_dice_loss(y_true, y_pred):
    return -tf.math.log(dice_coef(y_true, y_pred))

def iou(y_true, y_pred, threshold=0.5):                                                             
    y_true = tf.reshape(y_true, [-1])                                                               
    y_true = tf.cast(y_true, tf.float32)                                                            
    y_pred = tf.cast(y_pred > threshold, tf.float32)                                                 
    y_pred = tf.reshape(y_pred, [-1])                                                               
    intersection = tf.reduce_sum(y_true * y_pred)                                                     
    union = tf.reduce_sum(tf.cast(y_true + y_pred > 0, tf.float32))                                 
    return intersection / union

def weighted_loss(y_true, y_pred):
    """Binary Crossentropy + Dice loss with weighting"""
    alpha = 0.3  # Increase weight of positive pixels
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = 1 - (2 * tf.reduce_sum(y_true * y_pred) + 1e-6) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-6)
    return alpha * bce + (1 - alpha) * dice  # Weighted combination