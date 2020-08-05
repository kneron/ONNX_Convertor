import onnx
import argparse
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

from tools import helper
import onnx_vs_onnx as onnx_tester

def compare_results(results_a, results_b):
  """ compare onnx model inference results
      calculate basic statistical values
  results: results from inference multiple times
  returns: list of basic statistical values
  """
  # input results data can be of nonuniform shape
  # get flatten data to compare
  ra_flat = helper.flatten_with_depth(results_a, 0)
  rb_flat = helper.flatten_with_depth(results_b, 0)
  shape_a = [item[1] for item in ra_flat]
  shape_b = [item[1] for item in rb_flat]
  assert shape_a == shape_b, 'two results data shape doesn\'t match'
  ra_raw = [item[0] for item in ra_flat]
  rb_raw = [item[0] for item in rb_flat]

  # the statistical values
  max_rel_diff = 0  # defined to be max( { abs(diff)/max(abs(ra), abs(rb) ) } ) 
  max_abs_diff = 0  # defined to be max( { abs(ra-rb) } )
  mean_rel_diff = 0 
  mean_abs_diff = 0
  std_rel_diff = 0
  std_abs_diff = 0
  acc_with_diff_precision = []
  rel_diff = []    
  abs_diff_percentiles = []  # rel_diff percentiles
  rel_diff_percentiles = []  # abs_diff precentiles

  raw_diff = [ra_raw[i]-rb_raw[i] for i in range(len(ra_raw))]
  abs_diff = [abs(num) for num in raw_diff]
  for i in range(len(ra_raw)):
    divider = max([abs(ra_raw[i]), abs(rb_raw[i])])
    val = abs_diff[i]/divider if divider != 0 else 0
    rel_diff.append(val)

  max_rel_diff = max(rel_diff)
  max_abs_diff = max(abs_diff)
  mean_rel_diff = np.average(rel_diff)
  mean_abs_diff = np.average(abs_diff)
  std_rel_diff = np.std(rel_diff)
  std_abs_diff = np.std(abs_diff)

  # calculate accuracy with different precison
  for digit in range(8):
    correct = 0
    for i in range(len(ra_raw)):
      if format(ra_raw[i], '.'+str(digit)+'f')\
        == format(rb_raw[i], '.'+str(digit)+'f'):
        correct += 1
    acc_with_diff_precision.append([digit, float(format(correct/len(ra_raw), '.3f'))])

  # analyze rel_diff distribution
  rel_diff.sort()
  abs_diff.sort()
  for i in range(20):
    rel_diff_percentiles.append(['{}%'.format(i*5), rel_diff[int((i/20)*len(rel_diff))]])
    abs_diff_percentiles.append(['{}%'.format(i*5), abs_diff[int((i/20)*len(abs_diff))]])

  results = [
    ['max_rel_diff', max_rel_diff],
    ['max_abs_diff', max_abs_diff],
    ['mean_rel_diff', mean_rel_diff],
    ['mean_abs_diff', mean_abs_diff],
    ['std_rel_diff', std_rel_diff],
    ['std_abs_diff', std_abs_diff],
    ['acc_with_diff_precision', acc_with_diff_precision],
    ['rel_diff_percentiles', rel_diff_percentiles],
    ['abs_diff_percentiles', abs_diff_percentiles]
  ]
  
  return results

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='test model optimization results')
  
  parser.add_argument('dir', type=str, help='the directory that stores onnx models')
  parser.add_argument('ending1', type=str, help='model file name ending(eg, .onnx)')
  parser.add_argument('ending2', type=str, help='opt model file name ending(eg. _opt.onnx)')
  parser.add_argument('out_file', type=str, help='output csv file name')
  parser.add_argument('-p', '--plot', default='N', help='get plots (Y/N)')
  parser.add_argument('-i', '--iter_times', default=10, type=int, help='inference times')

  args = parser.parse_args()

  old_models_paths = glob.glob(args.dir+'*'+args.ending1)
  new_models_paths = glob.glob(args.dir+'*'+args.ending2)

  stats_table = [[
      'Model',
    'max_rel_diff',
    'max_abs_diff',
    'mean_rel_diff',
    'mean_abs_diff',
    'std_rel_diff',
    'std_abs_diff',
    'acc_with_diff_precision',
    'rel_diff_percentiles',
    'abs_diff_percentiles'
      ]]

  for new_model_path in new_models_paths:
    old_model_path = new_model_path[:-len(args.ending2)] + args.ending1
    if old_model_path not in old_models_paths:
      continue

    # run inference
    results_a, results_b = onnx_tester.onnx_model_results(old_model_path, new_model_path, total_times=args.iter_times)

    # compare inference results
    comparision = compare_results(results_a, results_b)

    new_line = [old_model_path.split('/')[-1]]
    for item in comparision:
      new_line.append(item[1])

    stats_table.append(new_line)
  
  # try to read existing file
  old_stats_table = []
  try:
    old_file = open(args.out_file, 'r')
    reader = csv.reader(old_file)
    old_header = reader.__next__()
    for row in reader:
      old_stats_table.append(row)
    old_file.close()
  except:
    pass
  
  # compare and merge possible old stat data file with new stat data file
  header = stats_table[0]
  stats_table = stats_table[1:]
  new_model_names = set([item[0] for item in stats_table])
  for row in old_stats_table:
    if row[0] not in new_model_names:
      stats_table.append(row)
  stats_table.insert(0, header)

  # write a new stat data file, overwrite old file
  new_file = open(args.out_file, 'w', newline='')
  writer = csv.writer(new_file)
  for row in stats_table:
    writer.writerow(row)
  new_file.close() 

  # make some plots
  if args.plot == 'Y':
    if len(stats_table) < 2:
      exit(0)

    sample_table = stats_table[1:] if len(stats_table) < 6 else stats_table[1:6]

    max_rel_diffs = [round(float(item[1]), 2) for item in stats_table[1:]]
    plt.hist(max_rel_diffs, bins=15)
    plt.title('Max Relavtive Difference Histogram')
    plt.xlabel('Max Relative Difference')
    plt.ylabel('Counts')
    plt.savefig('max_rel_diff_hist.png')
    plt.close()

    max_abs_diffs = [round(float(item[2]), 2) for item in stats_table[1:]]
    plt.hist(max_abs_diffs, bins=15)
    plt.title('Max Absolute Difference Histogram')
    plt.xlabel('Max Absolute Difference')
    plt.ylabel('Counts')
    plt.savefig('max_abs_diff_hist.png')
    plt.close()
    
    for line in sample_table:
      model_name = line[0]
      percentiles = line[-2]
      x = [round(i*(1/len(percentiles)), 2) for i in range(len(percentiles))]
      y = [ele[1] for ele in percentiles]
      plt.plot(x, y, label=model_name)
    plt.title('Rel_diff Percentiles of Raw and Optimized Models')
    plt.xlabel('percentage')
    plt.ylabel('relative difference')
    plt.legend()
    plt.savefig('rel_diff_percentiles.png')
    plt.close()

    for line in sample_table:
      model_name = line[0]
      percentiles = line[-1]
      x = [round(i*(1/len(percentiles)), 2) for i in range(len(percentiles))]
      y = [ele[1] for ele in percentiles]
      plt.plot(x, y, label=model_name)
    plt.title('Abs_diff Percentiles of Raw and Optimized Models')
    plt.xlabel('percentage')
    plt.ylabel('absolute difference')
    plt.legend()
    plt.savefig('abs_diff_percentiles.png')
    plt.close()

    for line in sample_table:
      model_name = line[0]
      accuracies = line[-3]
      x = [acc[0] for acc in accuracies]
      y = [acc[1] for acc in accuracies]
      plt.plot(x, y, label=model_name)
    plt.title('Accuracies with Different Precisions')
    plt.xlabel('Decimals')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig('precisions.png')
    plt.close()


  


