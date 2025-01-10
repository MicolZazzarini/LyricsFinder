from cleaning import cleaning
from preprocessing import preprocessing
from modeling import run_modeling_phase
from training import train_model
from performance_evaluation import evaluate_performance


def main():
    

    print("1: Cleaning")
    input_path = r'..\dataset\lyrics.csv'
    max_samples_per_genre = 50
    target_per_class = {
        'Pop': 51,
        'Rock': 51,
        'Hip-Hop': 51,
        'Metal': 51
    }
    output_path = cleaning(input_path, max_samples_per_genre, target_per_class)
    
    
    print("2: Preprocessing")
    train_loader, test_loader = preprocessing(output_path)
    

    print("3: Modeling")
    model, optimizer, criterion = run_modeling_phase()

    
    print("4: Training")
    train_losses, val_losses, all_preds, all_labels = train_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion
    )
    print("Training completed.")


    print("5: Performance Evaluation")
    evaluate_performance(
        train_losses=train_losses,
        val_losses=val_losses,
        all_labels=all_labels,
        all_preds=all_preds
    )
    print("Evaluation completed.")
    
if __name__ == "__main__":
    main()

