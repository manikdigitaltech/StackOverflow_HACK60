import os
from google import genai
from google.genai import types

def get_multimodal_embedding(image_path: str) -> list[float]:
    """
    Reads an image (plot, graph, or diagram) on Windows 11 and passes 
    the raw binary data to the free gemini-embedding-2 VLM model.
    """
    # 1. Safety Checks
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: The image file was not found at '{image_path}'")
        
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("Error: GEMINI_API_KEY environment variable is missing on this machine.")

    # 2. Instantiate the official GenAI Client 
    # (Picks up the environment variable automatically)
    client = genai.Client()

    # 3. Open and read image bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Automatically identify if it's a PNG or JPEG
    ext = os.path.splitext(image_path)[-1].lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"

    print(f"Processing '{image_path}' -> Requesting VLM Embeddings...")

    # 4. Request the visual asset vector representation
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            )
        ]
    )

    # 5. Extract vector
    vector = response.embeddings[0].values
    return vector

# --- Direct execution test ---
if __name__ == "__main__":
    # Target visual asset from a research paper
    sample_asset = "computation-11-00052-g002-550.jpg"
    
    # Generate a placeholder image if the chart does not exist yet to prevent crashes
    if not os.path.exists(sample_asset):
        from PIL import Image
        img = Image.new('RGB', (400, 400), color = (73, 109, 137))
        img.save(sample_asset)
        print(f"[Notice] Created a placeholder mock image at: {sample_asset}")

    # try:
    #     # Run pipeline
    #     embeddings_vector = get_multimodal_embedding(sample_asset)
        
    #     print("\n=== Extraction Complete ===")
    #     print(f"Vector Coordinates (Dimension Count): {len(embeddings_vector)}")
    #     print(f"First 5 Values: {embeddings_vector[:]}      
    # except Exception as e:
    #     print(f"\nExecution Failed: {str(e)}")

    try:
        # Run pipeline
        embeddings_vector = get_multimodal_embedding(sample_asset)
        
        print("\n=== Extraction Complete ===")
        print(f"Vector Coordinates (Dimension Count): {len(embeddings_vector)}")
        
        # ----------------------------------------------------
        # CHANGED: Printing the entire raw list of 3,072 values
        # ----------------------------------------------------
        print("\n--- ENTIRE 3072 EMBEDDING VECTOR ---")
        print(embeddings_vector)
        print("------------------------------------\n")
        
    except Exception as e:
        print(f"\nExecution Failed: {str(e)}")