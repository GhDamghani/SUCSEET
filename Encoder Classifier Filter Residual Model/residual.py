from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import data
import numpy as np
import model_module


def main(config):
    np.random.seed(config.SEED)

    train_dataset, val_dataset = data.get_train_val_datasets(
        config.feat,
        config.cluster,
        config.logits_residual,
        config.timepoints,
        config.num_eeg_channels,
        config.BATCH_SIZE,
        config.EPOCHS,
        config.DEVICE,
        config.VALIDATION_RATIO,
        config.P_SAMPLE,
    )
    X_train, y_train, res_train = model_module.get_all_batches(train_dataset, True)
    X_test, y_test, res_test = model_module.get_all_batches(val_dataset, True)
    clf = QuadraticDiscriminantAnalysis(reg_param=0)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    acc_train = clf.score(X_train, y_train)
    print(acc, acc_train)

    logit_test = clf.predict_log_proba(X_test)
    logit_train = clf.predict_log_proba(X_train)

    logits_residual = np.zeros((config.cluster.shape[0], config.num_classes)).astype(
        np.float32
    )
    logits_residual[np.array(train_dataset.indices) + config.timepoints - 1] = (
        logit_train
    )
    logits_residual[np.array(val_dataset.indices) + config.timepoints - 1] = logit_test

    np.save(config.logits_residual_file, logits_residual)


if __name__ == "__main__":
    import config

    main(config)
