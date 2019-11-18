def set_bar(pbar, t, last_ess, ess, acceptance_ratio, resample_status):
    pbar.set_description("Step #: {:2d} | Last ESS: {:6.2f} | ESS: {:6.2f} "
                         "| Mutated: {:.1%} | {} |"
                         .format(t + 1, last_ess, ess, acceptance_ratio,
                                 resample_status))
    return pbar
