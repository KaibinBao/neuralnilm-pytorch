from __future__ import print_function, division, absolute_import
from copy import copy
import numpy as np
import torch
from sklearn import metrics as skmetrics

class MetricsAccumulator():
    def __init__(self, on_power_threshold, max_power):
        self.__mean_metric_names = [
            "average_error",
            "mean_absolute_error",
            "mean_squared_error",
            "relative_error_in_total_target_energy",
            "error_in_total_energy_assigned"
        ]
        self.__sum_metric_names = [
            "sum_abs_diff",
            "energy_error",
            "estimate",
            "target",
            "tp", "tn", "fp", "fn", "union",
            "tp_thres", "tn_thres", "fp_thres", "fn_thres",
            "mae_on", "mae_off",
            "mse_on", "mse_off",
        ]
        self.__sum_metric_results = [
            "sum_abs_diff",
            "energy_error"
        ]
        self.__counter_names = [
            "tp", "tn", "fp", "fn",
            "tp_thres", "tn_thres", "fp_thres", "fn_thres",
            "on", "off", "count", "calls"
        ]
        self.on_power_threshold = on_power_threshold
        self.max_power = max_power
        self.reset_accumulator()


    def reset_accumulator(self):
        self.accumulated = {k: 0.0 for k in self.__mean_metric_names}
        self.summed = {k: 0.0 for k in self.__sum_metric_names}
        self.counter = {k: 0 for k in self.__counter_names}


    def accumulate_metrics(self, true_y, pred_y, **kwargs):
        assert(true_y.shape == pred_y.shape)
        true_y = true_y.flatten()
        pred_y = pred_y.flatten()

        clipped_pred_y = np.clip(pred_y, 0, None)
        clipped_true_y = np.clip(true_y, 0, None)

        for k, v in kwargs.items():
            v_old = self.accumulated.setdefault(k, 0.0)
            self.accumulated[k] = v_old + v

        count = len(true_y)

        abs_diff = np.fabs(clipped_pred_y - clipped_true_y)
        sum_abs_diff = np.sum(abs_diff)
        square_diff = np.square(clipped_pred_y - clipped_true_y)
        sum_square_diff = np.sum(square_diff)
        signed_error = clipped_pred_y - clipped_true_y
        sum_signed_error = np.sum(signed_error)

        self.summed["r2_score"] = skmetrics.r2_score(clipped_true_y, clipped_pred_y)
        self.summed["standard_deviation_of_error"] = np.sqrt(np.sum(np.square( \
        	sum_signed_error - (sum_signed_error/count) )) / count)

        self.counter["calls"] += 1

        self.summed["sum_abs_diff"] += sum_abs_diff
        self.accumulated["mean_absolute_error"] += sum_abs_diff
        self.accumulated["mean_squared_error"] += sum_square_diff
        self.accumulated["average_error"] += sum_signed_error

        self.summed['estimate'] += np.sum(pred_y)
        self.summed['target'] += np.sum(true_y)

        self.summed["union"] += np.sum(np.maximum(clipped_pred_y, clipped_true_y))
        self.summed["tp"] += np.sum(np.minimum(clipped_pred_y, clipped_true_y))
        self.summed["fp"] += np.sum(np.clip(clipped_pred_y - clipped_true_y, 0, None))
        self.summed["fn"] += np.sum(np.clip(clipped_true_y - clipped_pred_y, 0, None))
        self.summed["tn"] += np.sum(np.minimum(self.max_power - clipped_pred_y, self.max_power - clipped_true_y))

        above_threshold_pred = np.greater_equal(clipped_pred_y, self.on_power_threshold)
        above_threshold_true = np.greater_equal(clipped_true_y, self.on_power_threshold)
        below_threshold_pred = np.less(clipped_pred_y, self.on_power_threshold)
        below_threshold_true = np.less(clipped_true_y, self.on_power_threshold)
        tp = np.sum(above_threshold_pred & above_threshold_true)
        tn = np.sum(below_threshold_pred & below_threshold_true)
        true_above_threshold_count = np.sum(above_threshold_true)
        true_below_threshold_count = (len(above_threshold_true)-true_above_threshold_count)
        self.counter["tp"] += tp
        self.counter["tn"] += tn
        self.counter["fn"] += true_above_threshold_count - tp
        self.counter["fp"] += true_below_threshold_count - tn

        self.summed["mae_on"] += np.sum(np.where(above_threshold_true, abs_diff, 0))
        self.summed["mae_off"] += np.sum(np.where(below_threshold_true, abs_diff, 0))
        self.summed["mse_on"] += np.sum(np.where(above_threshold_true, square_diff, 0))
        self.summed["mse_off"] += np.sum(np.where(below_threshold_true, square_diff, 0))
        self.counter["on"] += true_above_threshold_count
        self.counter["off"] += true_below_threshold_count

        above_threshold_pred_y = np.where(above_threshold_pred, clipped_pred_y, 0)
        above_threshold_true_y = np.where(above_threshold_true, clipped_true_y, 0)
        self.summed["tp_thres"] += np.sum(np.minimum(above_threshold_pred_y, above_threshold_true_y))
        self.summed["fp_thres"] += np.sum(np.clip(above_threshold_pred_y - above_threshold_true_y, 0, None))
        self.summed["fn_thres"] += np.sum(np.clip(above_threshold_true_y - above_threshold_pred_y, 0, None))
        self.summed["tn_thres"] += np.sum(np.minimum(self.max_power - above_threshold_pred_y, self.max_power - above_threshold_true_y))

        self.counter["count"] += count


    def finalize_metrics(self):
        result = {}

        count = self.counter["count"] # would be very strange if this is 0

        for k, v in self.accumulated.items():
            result[k] = float(v / count)

        for k in self.__sum_metric_results:
            result[k] = float(self.summed[k])

        tp = float(self.counter["tp"])
        fp = float(self.counter["fp"])
        tn = float(self.counter["tn"])
        fn = float(self.counter["fn"])

        predicted_positives = tp + fp
        precision = float(tp / predicted_positives) if predicted_positives != 0.0 else np.nan
        result["precision_score"] = precision

        condition_positives = tp + fn
        recall = float(tp / condition_positives) if condition_positives != 0.0 else np.nan
        result["recall_score"] = recall

        divisor = precision + recall
        f1 = float( (2.0 * precision * recall) / divisor ) if divisor != 0.0 else np.nan
        result["f1_score"] = f1

        divisor = tp + fp + tn + fn
        accuracy = float((tp + tn) / divisor) if divisor != 0.0 else np.nan
        result["accuracy_score"] = accuracy

        condition_negatives = tn + fp
        specificity = float(tn / condition_negatives) if condition_negatives != 0.0 else np.nan
        result["specificity_score"] = specificity

        predicted_negatives = tn + fn
        npv = float(tn / predicted_negatives) if predicted_negatives != 0.0 else np.nan
        result["npv_score"] = npv

        condition_positives = tp + fn
        condition_negatives = tn + fp
        balanced_accuracy = float((tp) / condition_positives) if condition_positives != 0.0 else np.nan
        balanced_accuracy += float((tn) / condition_negatives) if condition_negatives != 0.0 else np.nan
        balanced_accuracy /= 2
        result["balanced_accuracy_score"] = balanced_accuracy

        divisor = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
        divisor = np.sqrt(divisor) if divisor >= 0.0 else np.nan
        mcc = float( ((tp*tn)-(fp*fn))/divisor ) if divisor != 0.0 else np.nan
        result["mcc_score"] = mcc

        tp = self.summed["tp"]
        fp = self.summed["fp"]
        tn = self.summed["tn"]
        fn = self.summed["fn"]

        intersection = tp
        union = self.summed["union"]
        result["match_rate"] = float(intersection / union) if union != 0.0 else np.nan

        predicted_positives = tp + fp
        precision = float(tp / predicted_positives) if predicted_positives != 0.0 else np.nan
        result["precision_energy"] = precision

        divisor = tp + fn
        recall = float(tp / divisor) if divisor != 0.0 else np.nan
        result["recall_energy"] = recall

        divisor = result["precision_energy"] + result["recall_energy"]
        f1 = float( (2.0 * precision * recall) / divisor ) if divisor != 0.0 else np.nan
        result["f1_energy"] = f1

        divisor = tp + fp + tn + fn
        accuracy = float((tp + tn) / divisor) if divisor != 0.0 else np.nan
        result["accuracy_energy"] = accuracy

        condition_negatives = tn + fp
        specificity = float(tn / condition_negatives) if condition_negatives != 0.0 else np.nan
        result["specificity_energy"] = specificity

        predicted_negatives = tn + fn
        npv = float(tn / predicted_negatives) if predicted_negatives != 0.0 else np.nan
        result["npv_energy"] = npv

        condition_positives = tp + fn
        condition_negatives = tn + fp
        balanced_accuracy = float((tp) / condition_positives) if condition_positives != 0.0 else np.nan
        balanced_accuracy += float((tn) / condition_negatives) if condition_negatives != 0.0 else np.nan
        balanced_accuracy /= 2
        result["balanced_accuracy_energy"] = balanced_accuracy

        divisor = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
        divisor = np.sqrt(divisor) if divisor >= 0.0 else np.nan
        mcc = float( ((tp*tn)-(fp*fn))/divisor ) if divisor != 0.0 else np.nan
        result["mcc_energy"] = mcc

        num_samples = tp + fp + tn + fn
        factor = np.log(num_samples) if num_samples > 0.0 else np.nan
        l = num_samples * factor
        divisor = (tp+fp)*(tp+fn)
        factor = np.log(tp/divisor) if ((divisor != 0.0) and (tp > 0.0)) else np.nan
        ltp = tp * factor
        divisor = (fp+tp)*(fp+tn)
        factor = np.log(fp/divisor) if ((divisor != 0.0) and (fp > 0.0)) else np.nan
        lfp = fp * factor
        divisor = (fn+tp)*(fn+tn)
        factor = np.log(fn/divisor) if ((divisor != 0.0) and (fn > 0.0)) else np.nan
        lfn = fn * factor
        divisor = (tn+fp)*(tn+fn)
        factor = np.log(tn/divisor) if ((divisor != 0.0) and (tn > 0.0)) else np.nan
        ltn = tn * factor
        condition_positives = tp + fn
        condition_negatives = tn + fp
        factor = np.log(condition_positives/num_samples) if ((num_samples != 0.0) and (condition_positives > 0.0)) else np.nan
        lp = condition_positives * factor
        factor = np.log(condition_negatives/num_samples) if ((num_samples != 0.0) and (condition_negatives > 0.0)) else np.nan
        ln = condition_negatives * factor
        divisor = l + lp + ln
        proficiency = ((l+ltp+lfp+lfn+ltn)/divisor) if divisor != 0.0 else np.nan
        result["proficiency_energy"] = proficiency

        tp = self.summed["tp_thres"]
        fp = self.summed["fp_thres"]
        tn = self.summed["tn_thres"]
        fn = self.summed["fn_thres"]

        predicted_positives = tp + fp
        precision = float(tp / predicted_positives) if predicted_positives != 0.0 else np.nan
        result["precision_energy_on"] = precision

        divisor = tp + fn
        recall = float(tp / divisor) if divisor != 0.0 else np.nan
        result["recall_energy_on"] = recall

        divisor = result["precision_energy_on"] + result["recall_energy_on"]
        f1 = float( (2.0 * precision * recall) / divisor ) if divisor != 0.0 else np.nan
        result["f1_energy_on"] = f1

        divisor = tp + fp + tn + fn
        accuracy = float((tp + tn) / divisor) if divisor != 0.0 else np.nan
        result["accuracy_energy_on"] = accuracy

        condition_negatives = tn + fp
        specificity = float(tn / condition_negatives) if condition_negatives != 0.0 else np.nan
        result["specificity_energy_on"] = specificity

        predicted_negatives = tn + fn
        npv = float(tn / predicted_negatives) if predicted_negatives != 0.0 else np.nan
        result["npv_energy_on"] = npv

        condition_positives = tp + fn
        condition_negatives = tn + fp
        balanced_accuracy = float((tp) / condition_positives) if condition_positives != 0.0 else np.nan
        balanced_accuracy += float((tn) / condition_negatives) if condition_negatives != 0.0 else np.nan
        balanced_accuracy /= 2
        result["balanced_accuracy_energy_on"] = balanced_accuracy

        divisor = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
        divisor = np.sqrt(divisor) if divisor >= 0.0 else np.nan
        mcc = float( ((tp*tn)-(fp*fn))/divisor ) if divisor != 0.0 else np.nan
        result["mcc_energy_on"] = mcc

        num_samples = tp + fp + tn + fn
        factor = np.log(num_samples) if num_samples > 0.0 else np.nan
        l = num_samples * factor
        divisor = (tp+fp)*(tp+fn)
        factor = np.log(tp/divisor) if ((divisor != 0.0) and (tp > 0.0)) else np.nan
        ltp = tp * factor
        divisor = (fp+tp)*(fp+tn)
        factor = np.log(fp/divisor) if ((divisor != 0.0) and (fp > 0.0)) else np.nan
        lfp = fp * factor
        divisor = (fn+tp)*(fn+tn)
        factor = np.log(fn/divisor) if ((divisor != 0.0) and (fn > 0.0)) else np.nan
        lfn = fn * factor
        divisor = (tn+fp)*(tn+fn)
        factor = np.log(tn/divisor) if ((divisor != 0.0) and (tn > 0.0)) else np.nan
        ltn = tn * factor
        condition_positives = tp + fn
        condition_negatives = tn + fp
        factor = np.log(condition_positives/num_samples) if ((num_samples != 0.0) and (condition_positives > 0.0)) else np.nan
        lp = condition_positives * factor
        factor = np.log(condition_negatives/num_samples) if ((num_samples != 0.0) and (condition_negatives > 0.0)) else np.nan
        ln = condition_negatives * factor
        divisor = l + lp + ln
        proficiency = ((l+ltp+lfp+lfn+ltn)/divisor) if divisor != 0.0 else np.nan
        result["proficiency_energy_on"] = proficiency

        sum_estimate = self.summed['estimate']
        sum_target = self.summed['target']

        error_in_total_energy_assigned = np.fabs(float(sum_estimate - sum_target))
        result["error_in_total_energy_assigned"] = error_in_total_energy_assigned

        result["deviation"] = float(error_in_total_energy_assigned / sum_target)

        relative_error_in_total_target_energy = float( \
            (sum_estimate - sum_target) / sum_target )
        result["relative_error_in_total_target_energy"] = relative_error_in_total_target_energy

        result["energy_error"] = float(self.summed["sum_abs_diff"] / sum_target)

        if self.counter["calls"] == 1:
        	result["r2_score"] = self.summed["r2_score"]
        	result["standard_deviation_of_error"] = self.summed["standard_deviation_of_error"]
        else:
        	result["r2_score"] = np.nan
        	result["standard_deviation_of_error"] = np.nan

        result["fraction_of_energy_explained"] = float(sum_estimate / sum_target)

        result["normalized_mean_absolute_error"] = float(result["mean_absolute_error"] / sum_target)
        result["normalized_mean_squared_error"] = float(result["mean_squared_error"] / sum_target)

        result["mean_absolute_error_on"] = float(self.summed["mae_on"] / self.counter["on"]) if self.counter["on"] != 0.0 else np.nan
        result["mean_absolute_error_off"] = float(self.summed["mae_off"] / self.counter["off"]) if self.counter["off"] != 0.0 else np.nan
        result["mean_squared_error_on"] = float(self.summed["mse_on"] / self.counter["on"]) if self.counter["on"] != 0.0 else np.nan
        result["mean_squared_error_off"] = float(self.summed["mse_off"] / self.counter["off"]) if self.counter["off"] != 0.0 else np.nan

        return result


    def run_metrics(self, y_true, y_pred, mains):
        # Truncate
        n = min(len(y_true), len(y_pred))
        y_true = y_true[:n]
        y_pred = y_pred[:n]

        self.reset_accumulator()
        self.accumulate_metrics(y_true, y_pred)
        result = self.finalize_metrics()

        # For total energy correctly assigned
        denominator = 2 * np.sum(mains)
        sum_abs_diff = result['sum_abs_diff']
        total_energy_correctly_assigned = 1 - (sum_abs_diff / denominator)
        total_energy_correctly_assigned = float(total_energy_correctly_assigned)
        result['total_energy_correctly_assigned'] = total_energy_correctly_assigned

        return result


    def calculate_pr_curve(accumulated_pr, max_target_power, true_y, pred_y, num_thresholds):
        assert(true_y.shape == pred_y.shape)
        true_y = np.clip(true_y, 0, None).flatten()
        pred_y = np.clip(pred_y, 0, None).flatten()
        num_y = true_y.shape[0]

        tp = accumulated_pr["tp"]
        fp = accumulated_pr["fp"]
        tn = accumulated_pr["tn"]
        fn = accumulated_pr["fn"]

        for i, thres in enumerate(np.linspace(0, max_target_power, num=num_thresholds), 0):
            threshold_pred_y = (pred_y >= thres)
            threshold_true_y = (true_y >= thres)
            threshold_true_y_neg = np.logical_not(threshold_true_y)
            tp_i = np.sum(threshold_pred_y & threshold_true_y)
            tp[i] += int(tp_i)
            fp_i = np.sum(threshold_pred_y & threshold_true_y_neg)
            fp[i] += int(fp_i)
            fn[i] += int(np.sum(threshold_true_y) - tp_i)
            tn[i] += int(np.sum(threshold_true_y_neg) - fp_i)

        return accumulated_pr


    def calculate_pr_curve_torch(accumulated_pr, max_target_power, true_y, pred_y, num_thresholds):
        assert(true_y.shape == pred_y.shape)
        true_y = torch.from_numpy(true_y).clamp_(min=0).view(-1)
        pred_y = torch.from_numpy(pred_y).clamp_(min=0).view(-1)
        num_y = true_y.shape[0]

        tp = accumulated_pr["tp"]
        fp = accumulated_pr["fp"]
        tn = accumulated_pr["tn"]
        fn = accumulated_pr["fn"]

        for i, thres in enumerate(np.linspace(0, max_target_power, num=num_thresholds), 0):
            threshold_pred_y = torch.ge(pred_y, thres)
            threshold_true_y = torch.ge(true_y, thres)
            threshold_true_y_neg = 1 - threshold_true_y
            tp_i = torch.sum(threshold_pred_y * threshold_true_y)
            tp[i] += tp_i
            fp_i = torch.sum(threshold_pred_y * threshold_true_y_neg)
            fp[i] += fp_i
            fn[i] += torch.sum(threshold_true_y) - tp_i
            tn[i] += torch.sum(threshold_true_y_neg) - fp_i

        return accumulated_pr

