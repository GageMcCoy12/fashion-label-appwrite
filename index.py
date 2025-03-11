import os
import base64
import json
import requests
from PIL import Image
import io

def encode_image(image_base64):
    """Process base64 image, resize to 512x512, and re-encode to base64."""
    try:
        img_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 512x512 maintaining aspect ratio
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        # Create new image with padding
        new_img = Image.new('RGB', (512, 512), (255, 255, 255))  # White background
        offset = ((512 - img.size[0]) // 2, (512 - img.size[1]) // 2)
        new_img.paste(img, offset)
        
        # Convert back to base64
        buffer = io.BytesIO()
        new_img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def main(req, res):
    """Analyze clothing items using GPT-4 Vision API in an Appwrite Function."""
    # Fetch OpenAI API Key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return res.json({'error': 'OPENAI_API_KEY not found in environment variables'}, 500)
    
    # Get request data from Appwrite function input
    try:
        data = json.loads(os.getenv("APPWRITE_FUNCTION_DATA", "{}"))
        if not data or 'images' not in data:
            return res.json({'error': 'No images provided in request body'}, 400)
        
        base64_images = data['images']
        if not isinstance(base64_images, list):
            base64_images = [base64_images]

        # Process and resize images
        processed_images = [encode_image(img) for img in base64_images]

        system_prompt = """You are a fashion expert analyzing clothing items in images.
        For each item, identify:
        1. Type of clothing/accessory
        2. Brand (if visible or recognizable, otherwise suggest a similar brand)
        3. Color (be specific with shades)
        4. Material (if visible, otherwise suggest the most likely material)
        5. Aesthetic/style (e.g., casual, formal, streetwear)
        6. Extra details (specific item or close alternative)

        Format response as a JSON array with:
        {
            "type": "",
            "brand": "",
            "color": "",
            "material": "",
            "aesthetic": "",
            "extra_details": "",
            "confidence": 0.0
        }
        
        Return results in the same order as provided images. Never use 'unknown' - suggest alternatives instead."""

        # Construct API request content
        content = [{"type": "text", "text": f"Analyze {len(processed_images)} clothing items and return structured JSON."}]
        for base64_image in processed_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        # OpenAI GPT-4 Vision API Call
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
            "max_tokens": 2000
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_data = response.json()

        if 'error' in response_data:
            return res.json({'error': response_data['error']}, 500)
        
        # Extract JSON response
        response_content = response_data['choices'][0]['message']['content']

        # Cleanup and parse JSON
        if "```json" in response_content:
            response_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            response_content = response_content.split("```")[1].split("```")[0].strip()
        
        analyses = json.loads(response_content)
        if not isinstance(analyses, list):
            analyses = [analyses]

        # Add confidence scores
        for analysis in analyses:
            analysis["confidence"] = 0.9

        return res.json({'success': True, 'analyses': analyses})

    except Exception as e:
        return res.json({'error': str(e)}, 500)
