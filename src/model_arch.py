enhangcing_params = {    
    # (in_channels, out_channels, kernel_size, stride, padding)
    "encoder":
        [[  0,  32, (5, 2), (2, 1), (1, 1)],
         ( 32,  64, (5, 2), (2, 1), (2, 1)),
         ( 64, 128, (5, 2), (2, 1), (2, 1))],
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    "decoder":
        [(256,  64, (5, 2), (2, 1), (2, 0), (1, 0)),
         (128,  32, (5, 2), (2, 1), (2, 0), (1, 0)),
         [ 64,   0, (5, 2), (2, 1), (1, 0), (0, 0), True]]
}

phase_d_params = [
    # (in_channels, out_channels, kernel_size, stride, padding, is_last)
    [ -1,   8, (5, 5), (2, 2), (0, 0), False],
    (  8,  16, (5, 5), (2, 2), (0, 0), False),
    ( 16,  32, (5, 5), (2, 2), (0, 0), False),
    ( 32,  64, (5, 5), (2, 2), (0, 0), False),
    ( 64, 128, (5, 5), (2, 2), (0, 0), False),
    (128, 128, (5, 5), (2, 2), (0, 0), True)
]