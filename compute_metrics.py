from argparse import ArgumentParser
from pathlib import Path


from protscape.ensemble import compute_rmwd, compute_pca_dist, plot_pca_space, compute_rmsf_corr, plot_rmsf_overlay
import json


def main(args):

    save_dir = args.output_dir / args.system
    save_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "system": args.system,
        "ref_topo": str(args.ref_topo),
        "ref_traj": str(args.ref_traj),
        "pred_topo": str(args.pred_topo),
        "pred_traj": str(args.pred_traj),
        "align_sel": args.align_sel,
        "point_sel": args.point_sel,
    }
    
    rmwd = compute_rmwd(args.ref_topo, args.ref_traj,
                        args.pred_topo, args.pred_traj,
                        align_to=args.ref_topo,
                        align_sel=args.align_sel,
                        point_sel=args.point_sel)
    
    output["rmwd_metrics"] = rmwd

    rmsf_corr = compute_rmsf_corr(args.ref_topo, args.ref_traj,
                                args.pred_topo, args.pred_traj,
                                align_to=args.ref_topo,
                                align_sel=args.align_sel,
                                point_sel=args.point_sel)

    output["rmsf_corr_metrics"] = {
        'rmsf_corr': rmsf_corr['corr'],
        'rmsf_corr_sq': rmsf_corr['corr_sq'],

    }

    fig = plot_rmsf_overlay(rmsf_corr)

    fig_path = save_dir / f"{args.system}_rmsf_overlay.png"
    fig.savefig(fig_path, dpi=300)


    for pca_mode in ["pool", "ref"]:

        pca_w2 = compute_pca_dist(args.ref_topo, args.ref_traj,
                                args.pred_topo, args.pred_traj,
                                align_to=args.ref_topo,
                                align_sel=args.align_sel,
                                point_sel=args.point_sel,
                                pca_mode=pca_mode, k=2,
                                metric="gaussian_w2")

        output[f"{pca_mode}_pca_metrics"] = {
            "w2": pca_w2['w2'],
            "w2_sq": pca_w2['w2_sq'],
            "k": pca_w2['k'],
            "features": pca_w2['mode'],
            "pca_mode": pca_w2['pca_mode'],
            "metric": pca_w2['metric'],
        }

        fig = plot_pca_space(pca_w2, show_gaussian=False)

        fig_path = save_dir / f"{args.system}_{pca_mode}_pca_space.png"
        fig.savefig(fig_path, dpi=300)


    json_output_path = save_dir / f"{args.system}_metrics.json"
    with open(json_output_path, "w") as f:
        json.dump(output, f, indent=4)



if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--ref_topo", type=Path, required=True, help="Path to the reference topology file.")
    parser.add_argument("--ref_traj", type=Path, required=True, help="Path to the reference trajectory file.")

    parser.add_argument("--pred_topo", type=Path, required=True, help="Path to the predicted topology file.")
    parser.add_argument("--pred_traj", type=Path, required=True, help="Path to the predicted trajectory file.")

    parser.add_argument("--align_sel", default="protein and name CA", type=str, help="Selection string for alignment.")
    parser.add_argument("--point_sel", default="protein and name CA", type=str, help="Selection string for metric calculation.")

    parser.add_argument("--system", type=str, default="default", help="Name of the system being evaluated.")
    parser.add_argument("--output_dir", type=Path, default=Path("./metrics_output"), help="Directory to save the output metrics and plots.")


    args = parser.parse_args()

    main(args)