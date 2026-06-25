import csv, math
import tensorflow as tf

#============== Shcedule LR ==========================
def scd_loss(student_logits, teacher_logits, labels, temperature):
    """Sample Confidence Distillation (Ec. 4-6).
    """
    t = temperature
    teacher_probs = tf.nn.softmax(teacher_logits / t, axis=1)   # [B, C]
    student_probs = tf.nn.softmax(student_logits / t, axis=1)   # [B, C]

    p_t_max = tf.reduce_max(teacher_probs, axis=1)              # [B]
    idx = tf.stack([tf.range(tf.shape(labels)[0]), labels], axis=1)
    p_s_true = tf.gather_nd(student_probs, idx)                 # [B]

    bt = tf.stack([p_t_max, 1.0 - p_t_max], axis=1)             # [B, 2]
    bs = tf.stack([p_s_true, 1.0 - p_s_true], axis=1)           # [B, 2]

    eps = 1e-7
    kl = tf.reduce_sum(bt * (tf.math.log(bt + eps) - tf.math.log(bs + eps)), axis=1)
    return tf.reduce_mean(kl) * (t ** 2)


def mcd_loss(student_logits, teacher_logits, labels, temperature):
    """Masked Correlation Distillation (Ec. 8-10), con máscara M_ge (>=).
    """
    t = temperature
    B = tf.shape(teacher_logits)[0]

    idx = tf.stack([tf.range(B), labels], axis=1)
    z_true = tf.gather_nd(teacher_logits, idx)                 # [B]
    z_true = tf.expand_dims(z_true, axis=1)                    # [B, 1]

    mask = teacher_logits >= z_true                            # [B, C] bool
    neg_inf = tf.constant(-1e9, dtype=teacher_logits.dtype)
    t_masked = tf.where(mask, neg_inf, teacher_logits)
    s_masked = tf.where(mask, neg_inf, student_logits)

    teacher_probs = tf.nn.softmax(t_masked / t, axis=1)        # [B, C]
    student_log_probs = tf.nn.log_softmax(s_masked / t, axis=1)

    eps = 1e-7
    kl = tf.reduce_sum(
        teacher_probs * (tf.math.log(teacher_probs + eps) - student_log_probs),
        axis=1,
    )
    return tf.reduce_mean(kl) * (t ** 2)

def rld_loss(student_logits, teacher_logits, labels_onehot, temperature=4.0, lambda_ce=1.0, lambda_kd=4.0):
    student_logits = tf.cast(student_logits, tf.float32)
    teacher_logits = tf.cast(teacher_logits, tf.float32)
    labels_onehot  = tf.cast(labels_onehot, tf.float32)

    labels = tf.argmax(labels_onehot, axis=1, output_type=tf.int32)  # [B]

    ce = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(   # one-hot, NO sparse
            labels=labels_onehot, logits=student_logits)
    )
    scd = scd_loss(student_logits, teacher_logits, labels, temperature)
    mcd = mcd_loss(student_logits, teacher_logits, labels, temperature)
    return lambda_ce*ce + lambda_ce * scd + lambda_kd * mcd

def mixup_loss(student_logits, teacher_logits,labels_onehot,temperature=2, lambda_ce=0.2, lambda_kd=0.8):
    student_logits = tf.cast(student_logits, tf.float32)
    teacher_logits = tf.cast(teacher_logits, tf.float32)
    labels_onehot  = tf.cast(labels_onehot, tf.float32)
    ce = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_onehot, logits=student_logits)
    )

    if temperature == 0:
        # Rescaled logits approach (Choi et al. 2023)
        std_s = tf.math.reduce_std(student_logits, axis=1, keepdims=True)
        std_t = tf.math.reduce_std(teacher_logits, axis=1, keepdims=True)
        student_scaled = student_logits / (std_s + 1e-8)
        teacher_scaled = teacher_logits / (std_t + 1e-8)
        teacher_probs = tf.nn.softmax(teacher_scaled)
        student_log_probs = tf.nn.log_softmax(student_scaled)
        kl = tf.reduce_mean(tf.reduce_sum(
            teacher_probs * (tf.math.log(teacher_probs + 1e-7) - student_log_probs), axis=1
        ))
        return kl
    else:
        # Original approach con temperatura fija
        teacher_probs = tf.nn.softmax(teacher_logits / temperature)
        student_log_probs = tf.nn.log_softmax(student_logits / temperature)
        kl = tf.reduce_mean(tf.reduce_sum(
            teacher_probs * (tf.math.log(teacher_probs + 1e-7) - student_log_probs), axis=1
        ))
        return lambda_ce*ce+lambda_kd*(kl * (temperature ** 2))
    
# ----------------- Distillation Loss ---------------------

def z_score(logits, temperature, eps=1e-7):
    """Z-score logit standardization (Algoritmo 1).
    Z(x) = (x - mean) / std / tau, por fila (cada muestra).
    """
    mean = tf.reduce_mean(logits, axis=1, keepdims=True)          # [B,1]
    std  = tf.math.reduce_std(logits, axis=1, keepdims=True)      # [B,1]
    return (logits - mean) / (std + eps) / temperature


def lskd_loss(student_logits, teacher_logits, labels_onehot,temperature=2.0, lambda_ce=0.1, lambda_kd=9.0):
    """KD vanilla + Z-score standardization (Algoritmo 2).
    Pérdida total = lambda_ce * CE + lambda_kd * tau^2 * KL.
    """
    student_logits = tf.cast(student_logits, tf.float32)
    teacher_logits = tf.cast(teacher_logits, tf.float32)
    labels_onehot  = tf.cast(labels_onehot, tf.float32)

    # ---- rama CE: logits CRUDOS del estudiante (sin Z-score) ----
    ce = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_onehot, logits=student_logits)
    )

    # ---- rama KD: logits ESTANDARIZADOS (Z-score) antes del softmax ----
    z_t = z_score(teacher_logits, temperature)
    z_s = z_score(student_logits, temperature)

    teacher_probs     = tf.nn.softmax(z_t, axis=1)
    student_log_probs = tf.nn.log_softmax(z_s, axis=1)

    eps = 1e-7
    kl = tf.reduce_mean(tf.reduce_sum(
        teacher_probs * (tf.math.log(teacher_probs + eps) - student_log_probs),
        axis=1,
    ))
    kd = kl * (temperature ** 2)

    return lambda_ce * ce + lambda_kd * kd