from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import pandas as pd
import os
import torch
import multiprocessing
from time import sleep
from nnunetv2.inference.export_prediction import export_prediction_from_logits, convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from nnunetv2.utilities.utils import empty_cache
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy


class nnUNetPredictorWithClassification(nnUNetPredictor):
    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = 2):
        """
        Overridden version that also collects classification outputs and saves subtype_results.csv
        """
        # Collect classification predictions for CSV
        classification_records = []

        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []

            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')
                properties = preprocessed['data_properties']

                #  Wait for export slots to free up
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                # Segmentation prediction
                prediction = self.predict_logits_from_preprocessed_data(data).cpu()

                # Classification prediction
                with torch.no_grad():
                    self.network.eval()
                    data_device = data.to(self.device)

                    if hasattr(self.network, 'encoder'):
                        features = self.network.encoder(data_device)
                        logits = self.network.classification_head(features)
                    else:
                        logits = self.network.classification_head(data_device)

                    cls_pred = int(torch.argmax(logits, dim=1).cpu().item())
                    case_name = os.path.basename(ofile) + '.nii.gz'
                    classification_records.append((case_name, cls_pred))

                # Export segmentation result
                if ofile is not None:
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape,
                            ((prediction, self.plans_manager,
                              self.configuration_manager, self.label_manager,
                              properties,
                              save_probabilities),)
                        )
                    )

                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')

            ret = [i.get()[0] for i in r]

        # Cleanup
        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()
        compute_gaussian.cache_clear()
        empty_cache(self.device)

        # Save classification results as CSV
        if classification_records:
            out_dir = os.path.dirname(ofile)
            df = pd.DataFrame(classification_records, columns=["Names", "Subtype"])
            df.to_csv(os.path.join(out_dir, "subtype_results.csv"), index=False)
            print(f"Saved classification results to {os.path.join(out_dir, 'subtype_results.csv')}")

        return ret
