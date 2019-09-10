try:
    from streamlit_demo.self_driving import app
except ModuleNotFoundError:
    import app

if __name__ == '__main__':
    app.main()
