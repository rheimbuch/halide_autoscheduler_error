import halide as hl
import math
from pprint import pprint

def focus_stack_pipeline():
    outputs = []
    start_w, start_h = 3000, 2000
    number_of_layers = 5
    layer_sizes = [[start_w, start_h]]    

    for i in range(0, number_of_layers):
            # Grab from prev layer
            w,h = layer_sizes[-1]
            layer_sizes.append([int(math.ceil(w/2.0)),int(math.ceil(h/2.0))])

    # Add last size in once more to get the 2nd top lap layer (gaussian) for
    # the energy/deviation split.
    layer_sizes.append(layer_sizes[-1])

    input = hl.ImageParam(hl.UInt(8), 3)
    input.dim(0).set_estimate(0, start_w)
    input.dim(1).set_estimate(0, start_h)
    input.dim(2).set_estimate(0, 3)

    lap_inputs = []
    max_energy_inputs = []

    for i in range(0,number_of_layers+1):
        lap_layer = hl.ImageParam(hl.Float(32), 3, "lap{}".format(i))
        lap_inputs.append(lap_layer)
        w,h = layer_sizes[i]
        lap_layer.dim(0).set_estimate(0, w)
        lap_layer.dim(1).set_estimate(0, h)
        lap_layer.dim(2).set_estimate(0, 3)

        if i == number_of_layers:
            # last (top - small) layer
            # Add the last laplacian (really direct from gaussian) layer
            # in twice. We output one maxed on entropies and one maxed on
            # deviations.
            lap_layer = hl.ImageParam(hl.Float(32), 3, "lap{}".format(i+1))
            lap_inputs.append(lap_layer)
            lap_layer.dim(0).set_estimate(0, w)
            lap_layer.dim(1).set_estimate(0, h)
            lap_layer.dim(2).set_estimate(0, 3)


            entropy_layer = hl.ImageParam(hl.Float(32), 2, "entroy{}".format(i))
            max_energy_inputs.append(entropy_layer)
            entropy_layer.dim(0).set_estimate(0, w)
            entropy_layer.dim(1).set_estimate(0, h)

            deviation_layer = hl.ImageParam(hl.Float(32), 2, "deviation{}".format(i))
            max_energy_inputs.append(deviation_layer)
            deviation_layer.dim(0).set_estimate(0, w)
            deviation_layer.dim(1).set_estimate(0, h)
        else:
            max_energy_layer = hl.ImageParam(hl.Float(32), 2, "max_energy{}".format(i))
            max_energy_inputs.append(max_energy_layer)
            max_energy_layer.dim(0).set_estimate(0, w)
            max_energy_layer.dim(1).set_estimate(0, h)

    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")
    hist_index = hl.Var('hist_index')
    clamped = f32(x, y, c, mirror(input, 3000, 2000))

    f = hl.Func("input32")
    f[x, y, c] = clamped[x, y, c]

    energy_outputs = []
    gaussian_layers = [f]

    laplacian_layers = []
    merged_laps = []

    for layer_num in range(0, number_of_layers):
        # Add the layer size in also
        w,h = layer_sizes[layer_num]

        start_layer = gaussian_layers[-1]

        # Blur the image
        gaussian_layer = gaussian(x, y, c, start_layer)

        # Grab next layer size
        # w,h = layer_sizes[layer_num+1]

        # Reduce the layer size and add it into the list
        next_layer = reduce_layer(x, y, c, gaussian_layer)
        gaussian_layers.append(next_layer)

        # Expand back up
        expanded = expand_layer(x, y, c, next_layer)

        # Generate the laplacian from the
        # original - blurred/reduced/expanded version
        laplacian_layer = laplacian(x, y, c, start_layer, expanded)
        laplacian_layers.append(laplacian_layer)

        # Calculate energies for the gaussian layer
        prev_energies = mirror(max_energy_inputs[layer_num], w, h)
        next_energies = region_energy(x, y, c, laplacian_layer)

        prev_laplacian = mirror(lap_inputs[layer_num], w, h)
        merged_energies = energy_maxes(x, y, c, prev_energies, next_energies)

        merged_lap = merge_laplacian(x, y, c, merged_energies, next_energies, prev_laplacian, laplacian_layer)

        energy_outputs.append([[w,h,True],merged_energies])
        merged_laps.append(merged_lap)

        # Add estimates
        next_layer.set_estimate(x, 0, w)
        next_layer.set_estimate(y, 0, h)
        next_layer.set_estimate(c, 0, 3)

    # Handle last layer differently
    w,h = layer_sizes[-1]

    # The next_lap is really just the last gaussian layer
    next_lap = gaussian_layers[-1]

    prev_entropy_laplacian = mirror(lap_inputs[-2], w, h)
    prev_entropy = mirror(max_energy_inputs[-2], w, h)
    next_entropy = entropy(x, y, c, next_lap, w, h, hist_index)
    merged_entropy = energy_maxes(x, y, c, prev_entropy, next_entropy)
    merged_lap_on_entropy = merge_laplacian(x, y, c, merged_entropy, next_entropy, prev_entropy_laplacian, next_lap)
    merged_laps.append(merged_lap_on_entropy)

    prev_deviation_laplacian = mirror(lap_inputs[-1], w, h)
    prev_deviation = mirror(max_energy_inputs[-1], w, h)
    next_deviation = deviation(x, y, c, next_lap)
    merged_deviation = energy_maxes(x, y, c, prev_deviation, next_deviation)
    merged_lap_on_deviation = merge_laplacian(x, y, c, merged_deviation, next_deviation, prev_deviation_laplacian, next_lap)
    merged_laps.append(merged_lap_on_deviation)

    energy_outputs.append([[w,h,True],merged_entropy])
    energy_outputs.append([[w,h,True],merged_deviation])


    print("NUM LAYERS: ", len(gaussian_layers), len(laplacian_layers), layer_sizes)
    
    # Add all of the laplacian layers to the output first
    i = 0
    for merged_lap in merged_laps:
        w,h = layer_sizes[i]
        mid = (i < (len(merged_laps) - 2))
        outputs.append([[w,h,False,mid], merged_lap])
        i += 1

    # Then energies
    for energy_output in energy_outputs:
        outputs.append(energy_output)

    new_outputs = []
    for size, output in outputs:
        w = size[0]
        h = size[1]
        gray = len(size) > 2 and size[2]
        mid = len(size) > 3 and size[3]

        if mid:
            uint8_output = output
        else:
            uint8_output = output

        uint8_output.set_estimate(x, 0, w)
        uint8_output.set_estimate(y, 0, h)
        if not gray:
            uint8_output.set_estimate(c, 0, 3)

        new_outputs.append([size, uint8_output])

    outputs = new_outputs

    print("OUTPUT LAYERS: ")
    pprint(outputs)

    output_funcs = [output for _, output in outputs]
    
    pipeline = hl.Pipeline(output_funcs)

    return {
        'pipeline': pipeline,
        'inputs': [input] + lap_inputs + max_energy_inputs
    }


def mkfunc(name, *imgs):
    return hl.Func('{}'.format(name))


def mirror(img, w, h):
    return hl.BoundaryConditions.mirror_interior(img)


def u8(x, y, c, img):
    out = mkfunc("u8", img)
    if img.dimensions() == 2:
        out[x, y] = hl.cast(hl.UInt(8), img[x, y])
    else:
        out[x, y, c] = hl.cast(hl.UInt(8), img[x, y, c])
    return out


def f32(x, y, c, img):
    out = mkfunc("f32", img)

    if img.dimensions() == 2:
        out[x, y] = hl.cast(hl.Float(32), img[x, y])
    else:
        out[x, y, c] = hl.cast(hl.Float(32), img[x, y, c])

    return out


def region_energy(x, y, c, img):
    _gray = gray(x, y, c, img)

    # use gaussian blur on squarred laplacian
    gray_squared = mkfunc('gray_sqrd', img)
    gray_squared[x,y] = (_gray[x,y] * _gray[x,y])
    return gaussian_1d(x, y, c, gray_squared)


def gray(x, y, c, img):
    gray = mkfunc("gray", img)
    # BGR
    gray[x, y] = 0.114*img[x,y,0] + 0.587*img[x,y,1] + 0.299*img[x,y,2]
    return gray


def gaussian(x, y, c, f):
    gaus_y = hl.Func("gaus_y")
    gaus_x = mkfunc("gaus", f)

    kernel = [0.05, 0.25, 0.4, 0.25, 0.05]

    gaus_y[x, y, c] = (kernel[0] * f[x, y - 2, c]) + \
                      (kernel[1] * f[x, y - 1, c]) + \
                      (kernel[2] * f[x, y, c]) + \
                      (kernel[3] * f[x, y + 1, c]) + \
                      (kernel[4] * f[x, y + 2, c])

    gaus_x[x, y, c] = (kernel[0] * gaus_y[x - 2, y, c]) + \
                      (kernel[1] * gaus_y[x - 1, y, c]) + \
                      (kernel[2] * gaus_y[x, y, c]) + \
                      (kernel[3] * gaus_y[x + 1, y, c]) + \
                      (kernel[4] * gaus_y[x + 2, y, c])

    return gaus_x


def gaussian_1d(x, y, c, f):
    gaus_y = hl.Func("gaus_y1d")
    gaus_x = mkfunc("gaus_x1d", f)

    kernel = [0.05, 0.25, 0.4 , 0.25, 0.05]

    gaus_y[x, y] = (kernel[0] * f[x, y-2]) + \
                        (kernel[1] * f[x, y-1]) + \
                        (kernel[2] * f[x, y]) + \
                        (kernel[3] * f[x, y+1]) + \
                        (kernel[4] * f[x, y+2])

    gaus_x[x, y] = (kernel[0] * gaus_y[x-2, y]) + \
                        (kernel[1] * gaus_y[x-1, y]) + \
                        (kernel[2] * gaus_y[x, y]) + \
                        (kernel[3] * gaus_y[x+1, y]) + \
                        (kernel[4] * gaus_y[x+2, y])

    return gaus_x


def reduce_layer(x, y, c, img):
    reduced = mkfunc("reduce", img)
    reduced[x, y, c] = img[x*2, y*2, c]
    return reduced


def expand_layer(x, y, c, img):
        expanded = hl.Func('expanded')
        expanded[x, y, c] = hl.select(((x % 2 == 0) & (y % 2 == 0)), img[x // 2, y // 2, c], 0.0)
        blurred = gaussian(x, y, c, expanded)
        expanded2 = mkfunc("expand", img)
        expanded2[x,y,c] = blurred[x,y,c] * 4.0
        return expanded2

def laplacian(x, y, c, original, gaussian):    
    laplacian = mkfunc("laplacian", original, gaussian)
    laplacian[x, y, c] = original[x, y, c] - gaussian[x, y, c]
    return laplacian


def energy_maxes(x, y, c, start_energy, next_energy):
    combined = mkfunc("energy_max", start_energy, next_energy)
    combined[x,y] = hl.max(start_energy[x,y], next_energy[x,y])
    return combined


def merge_laplacian(x, y, c, merged_energy, next_energy, prev_lap, next_lap):
    merged_lap = mkfunc('merged_lap', merged_energy, next_energy, next_lap, prev_lap)
    merged_lap[x,y,c] = hl.select(merged_energy[x,y] == next_energy[x,y],
                                    next_lap[x,y,c], prev_lap[x,y,c])
    return merged_lap


def entropy(x, y, c, img, w, h, hist_index):
    base_gray = gray(x, y, c, img)
    clamped_gray = mkfunc('clamped_gray', base_gray)
    clamped_gray[x,y] = hl.clamp(base_gray[x,y], 0, 255)
    u8_gray = u8(x, y, c, clamped_gray)

    probabilities = histogram(x, y, c, u8_gray, w, h, hist_index)

    r = hl.RDom([(-2, 5), (-2, 5)])

    levels = mkfunc('entropy', img)
    levels[x,y] = 0.0
    # Add in 0.00001 to prevent -Inf's
    levels[x,y] += base_gray[x + r.x, y + r.y] * hl.log(probabilities[u8_gray[x + r.x, y + r.y]]+0.00001)
    levels[x,y] = levels[x,y] * -1.0

    return levels


def deviation(x, y, c, img):
    _gray = gray(x, y, c, img)

    r = hl.RDom([(-2, 5), (-2, 5)])

    avg = mkfunc('avg', _gray)
    avg[x,y] = 0.0
    avg[x,y] += _gray[x + r.x, y + r.y]
    avg[x,y] = avg[x,y] / 25.0

    deviation = mkfunc('deviation', avg)
    deviation[x,y] = 0.0
    deviation[x,y] += (_gray[x + r.x, y + r.y] - avg[x,y]) ** 2
    deviation[x,y] = (deviation[x,y] / 25.0)

    return deviation


# Expects a u8 gray image
def histogram(x, y, c, img, w, h, hist_index):
    print("GET HIST ON: ", w, h)
    histogram = hl.Func("histogram")

    # Histogram buckets start as zero.
    histogram[hist_index] = 0

    # Define a multi-dimensional reduction domain over the input image:
    r = hl.RDom([(0, w), (0, h)])

    # For every point in the reduction domain, increment the
    # histogram bucket corresponding to the intensity of the
    # input image at that point.

    histogram[hl.Expr(img[r.x, r.y])] += 1
    histogram.set_estimate(hist_index, 0, 255)

    # Get the sum of all histogram cells
    r = hl.RDom([(0,255)])
    hist_sum = hl.Func('hist_sum')
    hist_sum[()] = 0.0 # Compute the sum as a 32-bit integer
    hist_sum[()] += histogram[r.x]

    # Return each histogram as a % of total color
    pct_hist = hl.Func('pct_hist')
    pct_hist[hist_index] = histogram[hist_index] / hist_sum[()]

    return histogram


def autoschedule(pipeline, autoscheduler_name, target, machine):
    hl.load_plugin('auto_schedule')
    pipeline.set_default_autoscheduler_name(autoscheduler_name)
    return pipeline.auto_schedule(target, machine)


if __name__ == "__main__":
    fs = focus_stack_pipeline()
    print("Autoscheduling with: Adams2019")
    autoschedule(fs['pipeline'], "Adams2019", hl.get_target_from_environment(), hl.MachineParams(4, 256*1024, 50))
    
    