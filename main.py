"""Run the Part 4 evaluator from inputs/manifest.csv."""
import argparse
from pathlib import Path


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=project / 'inputs/manifest.csv')
    parser.add_argument('--output', type=Path, default=project / 'outputs/current')
    parser.add_argument('--require-complete', action='store_true',
                        help='Reject missing references instead of saving partial development results.')
    parser.add_argument('--confusion', action='store_true',
                        help='Compare saved word predictions with independent heard-word labels.')
    parser.add_argument('--human-evaluation', action='store_true',
                        help='Evaluate saved model scores against genuine human reviews; no model inference.')
    args = parser.parse_args()
    if args.human_evaluation:
        if args.confusion or args.require_complete:
            parser.error('--human-evaluation runs separately from inference/confusion options')
        from evaluate_humans import main as evaluate_humans
        evaluate_humans()
        return
    if args.confusion:
        from text_methods import write_confusion_matrices
        write_confusion_matrices(args.manifest, args.output / 'scores.csv',
                                 project / 'answers/heard_words.csv', args.output)
        return
    from evaluate import evaluate_manifest
    evaluate_manifest(args.manifest, args.output, args.require_complete)


if __name__ == '__main__':
    main()
