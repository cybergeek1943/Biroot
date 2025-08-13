import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from functools import wraps
from data_generation import load_dataset
from tqdm import tqdm

c_val = 1
# dataset = load_dataset(f'binom_biroot_data\\binom_biroot_c_is_{c_val}.npz')
dataset = load_dataset(f'dag_biroot_c_is_{c_val}.npz')

# noinspection PyDefaultArgument
@wraps(sns.heatmap)
def plot(
colors: tuple = ('white', 'brown'),
color_bins: int = 256,
save_to: str = '',
**kwargs
) -> None:
    cmap = mcolors.LinearSegmentedColormap.from_list('cmap', colors, N=color_bins)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.heatmap(**kwargs, cmap=cmap, ax=ax)
    plt.tight_layout()

    # End
    if save_to:
        plt.savefig(save_to)
        plt.close(fig)


def save_binomial_batch():
    # noinspection PyDefaultArgument
    def config(defaults: dict = {}, **kwargs) -> dict:
        _ = defaults.copy()
        _.update(kwargs)
        return _

    # ================ Full View ================
    # Defaults
    fv_default_err_1 = config(
        cbar=False, vmax=1,
        xticklabels=1000, yticklabels=10,
    )
    fv_default_err_m = config(
        cbar=False, vmax=0.000002,
        xticklabels=1000, yticklabels=10,
        colors=('white', 'white', 'brown')
    )

    class FV:
        # 1 error
        configs_err_1 = [
            config(fv_default_err_1,
                   data=dataset[i], save_to=f'c_{c_val}_n_{i}_err_1'
            )
            for i in (3, 4, 5, 6)
        ]

        # machine error
        configs_err_m = [
            config(fv_default_err_m,
                   data=dataset[3][:, :4000], save_to=f'c_{c_val}_n_3_err_m'
            )
        ]
        configs_err_m.extend([
            config(fv_default_err_m,
                   data=dataset[i], save_to=f'c_{c_val}_n_{i}_err_m'
            )
            for i in (4, 5, 6)
        ])

    # ================ Part View ================
    # Defaults
    pv_default_err_1 = config(
        cbar=False, vmax=1,
        xticklabels=10, yticklabels=4,
    )
    pv_default_err_m = config(
        cbar=False, vmax=0.000002,
        xticklabels=10, yticklabels=4,
        colors=('white', 'white', 'brown')
    )

    class PV:
        # 1 error
        configs_err_1 = [
            config(pv_default_err_1,
                   data=dataset[i][:40, :100], save_to=f'pv_c_{c_val}_n_{i}_err_1'
            )
            for i in (3, 4, 5, 6)
        ]

        # Machine Error
        configs_err_m = [
            config(pv_default_err_m,
                data=dataset[3][16:56, :100], save_to=f'pv_c_{c_val}_n_3_err_m'
            ),
            config(pv_default_err_m,
                data=dataset[4][32:72, :100], save_to=f'pv_c_{c_val}_n_4_err_m'
            ),
            config(pv_default_err_m,
                data=dataset[5][60:100, :100], save_to=f'pv_c_{c_val}_n_5_err_m'
            ),
            config(pv_default_err_m,
                data=dataset[6][90:130, :100], save_to=f'pv_c_{c_val}_n_6_err_m'
            ),
        ]


    # Plot Configs
    selected_configs = [
        # FV.configs_err_1,
        FV.configs_err_m,
        # PV.configs_err_1,
        # PV.configs_err_m
    ]
    for cfgs in selected_configs:
        for cfg in tqdm(cfgs):
            plot(**cfg)


def save_gaussian_batch():
    # noinspection PyDefaultArgument
    def config(defaults: dict = {}, **kwargs) -> dict:
        _ = defaults.copy()
        _.update(kwargs)
        return _

    # ================ Full View ================
    # Defaults
    fv_default_err_1 = config(
        cbar=False, vmax=1,
        xticklabels=1000, yticklabels=5,
    )
    fv_default_err_m = config(
        cbar=False, vmax=0.00002,
        xticklabels=1000, yticklabels=5,
        colors=('white', 'white', 'brown')
    )

    class FV:
        # 1 error
        configs_err_1 = [
            config(fv_default_err_1,
                data=dataset[i], save_to=f'gaussian_c_{c_val}_n_{i}_err_1'
            )
            for i in (2, 3, 4, 5)
        ]

        # machine error
        configs_err_m = [
            config(fv_default_err_m,
                data=dataset[i], save_to=f'gaussian_c_{c_val}_n_{i}_err_m'
            )
            for i in (2, 3, 4, 5)
        ]

    # Plot Configs
    selected_configs = [
        FV.configs_err_1,
        FV.configs_err_m,
    ]
    for cfgs in selected_configs:
        for cfg in tqdm(cfgs):
            plot(**cfg)


def save_dag_batch():
    # noinspection PyDefaultArgument
    def config(defaults: dict = {}, **kwargs) -> dict:
        _ = defaults.copy()
        _.update(kwargs)
        return _

    # ================ Full View ================
    # Defaults
    fv_default_err_1 = config(
        cbar=False, vmax=1,
        xticklabels=1000, yticklabels=5,
    )
    fv_default_err_m = config(
        cbar=False, vmax=0.00002,
        xticklabels=200, yticklabels=5,
        colors=('white', 'white', 'brown')
    )

    class FV:
        # 1 error
        configs_err_1 = [
            config(fv_default_err_1,
                data=dataset[i], save_to=f'dag_c_{c_val}_n_{i}_err_1'
            )
            for i in (2, 3, 4, 5)
        ]

        # machine error
        configs_err_m = [
            config(fv_default_err_m,
                data=dataset[i][:, :2000], save_to=f'dag_c_{c_val}_n_{i}_err_m'
            )
            for i in (2, 3, 4, 5)
        ]

    # Plot Configs
    selected_configs = [
        # FV.configs_err_1,
        FV.configs_err_m,
    ]
    for cfgs in selected_configs:
        for cfg in tqdm(cfgs):
            plot(**cfg)



if __name__ == '__main__':
    # data = dataset[2]
    # plot(data=data[:,:], vmax=1, colors=('white', 'white', 'brown'))
    # # plot(data=data, vmax=1)
    # plt.show()


    # save_binomial_batch()
    # save_gaussian_batch()
    save_dag_batch()

