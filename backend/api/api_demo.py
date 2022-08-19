import uvicorn
from fastapi import FastAPI
import functions_api as fn

# Start API
app = FastAPI(title= 'API demo', 
              docs_url= '/docs', 
              redoc_url= '/redoc',
              description= 'API demostrativa',
              version = '0.1.0',
              root_path="/demoAPI/")


@app.get('/')
async def home():
    return 'API demo: estas en el home de la API'

@app.put('/upload')
async def upload(periodo: str, total_boletas: str):
    return fn.upload(periodo, total_boletas)
    
 
@app.get('/datos_temporales') 
async def datos_temporales():
    return fn.datos_temporales()

@app.get('/forecast')
async def forecast(steps: int = 15):
    return fn.forecast(steps)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8103)