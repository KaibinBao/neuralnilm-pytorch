from __future__ import print_function, division, absolute_import

from copy import copy
import numpy as np
import torch
import time
import tensorboardX

def SW_add_scalars2(self, main_tag, tag_scalar_dict, global_step=None):
    """Adds many scalar data to summary.
    Args:
        tag (string): Data identifier
        main_tag (string): The parent name for the tags
        tag_scalar_dict (dict): Key-value pair storing the tag and corresponding values
        global_step (int): Global step value to record
    Examples::
        writer.add_scalars('run_14h',{'xsinx':i*np.sin(i/r),
                                      'xcosx':i*np.cos(i/r),
                                      'arctanx': numsteps*np.arctan(i/r)}, i)
        # This function adds three values to the same scalar plot with the tag
        # 'run_14h' in TensorBoard's scalar section.
    """
    timestamp = time.time()
    fw_logdir = self.file_writer.get_logdir()
    for tag, scalar_value in tag_scalar_dict.items():
        fw_tag = fw_logdir + "/" + tag
        #fw_tag_full = fw_logdir + "/" + main_tag + "/" + tag
        if fw_tag in self.all_writers.keys():
            fw = self.all_writers[fw_tag]
        else:
            fw = tensorboardX.FileWriter(logdir=fw_tag)
            self.all_writers[fw_tag] = fw
        fw.add_summary(tensorboardX.summary.scalar(main_tag, scalar_value), global_step)
        #self.__append_to_scalar_dict(fw_tag_full, scalar_value, global_step, timestamp)
